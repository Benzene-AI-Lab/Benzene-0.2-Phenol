#!/usr/bin/env python3
"""
train_gpu.py —— 在 ModelScope A10 上训练

与 train.py 的区别：
    1. 自动用 cuda + bf16 混合精度（A10 是 Ampere，原生支持 bf16，不需要 GradScaler）
    2. 大 batch（1024），充分利用显存
    3. 断点续训（每 epoch 存一次，被掐断可以接着跑）
    4. 显存与时间监控，防止实例超时
    5. 训练完自动跑一次对弈抽样

用法：
    python train_gpu.py --data ../data/labels_100k.npz --scale alkane \\
        --epochs 30 --batch-size 512 --out ../weights/0.2-init.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from board import NUM_MOVES
from net import build as build_net
from train import LabelDataset, collate, evaluate, top1_accuracy


def pick_device(req: str) -> torch.device:
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale", default="alkane")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--auto-cpu", action="store_true",
                    help="检测到 CPU 时自动调整 batch / workers")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0,
                    help="数据全在内存，0 最快（多进程反而被 IPC 拖慢）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--out", default="../weights/0.2-init.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-hours", type=float, default=6.0,
                    help="超过这个时长就主动停止，防止实例被强杀丢进度")
    args = ap.parse_args()

    # CPU 上必须显式设置线程数，否则 PyTorch 默认只用一个物理核的数量
    if args.device != "cuda":
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(1)   # 单进程训练，interop 开多没用
        print(f"[线程] torch 使用 {n_threads} 线程（逻辑核 {os.cpu_count()}）")

    device = pick_device(args.device)
    if args.auto_cpu and device.type == "cpu":
        # CPU 上 batch 太大反而慢（缓存打不满），256 通常最优
        if args.batch_size > 256:
            args.batch_size = 256
        args.workers = 0      # 实测：workers=0 比 4 快 200 倍
        print(f"[自适应] CPU 模式：batch {args.batch_size}, workers {args.workers}")
    use_amp = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    use_scaler = args.amp == "fp16" and device.type == "cuda"

    print("=" * 68)
    print(f"设备      {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"显卡      {p.name}  {p.total_memory/1024**3:.1f} GB")
    print(f"精度      {args.amp}")
    print("=" * 68)

    ds = LabelDataset(args.data)
    train_ds = ds.subset(ds.train_idx)
    val_ds = ds.subset(ds.val_idx)

    pin = device.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, drop_last=True, pin_memory=pin,
        persistent_workers=args.workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate,
        num_workers=args.workers, pin_memory=pin,
    )

    net = build_net(args.scale).to(device)
    n_param = net.count_params()

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.03))

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    import math
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        floor = args.min_lr / args.lr
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_epoch = 1
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("val_loss", float("inf"))
        print(f"续训：从 epoch {start_epoch} 开始，best val {best_val:.4f}")

    print(f"参数      {n_param/1e6:.2f} M ({args.scale})")
    print(f"数据      {len(train_ds)} 训练 / {len(val_ds)} 验证")
    print(f"计划      {args.epochs} epoch × {steps_per_epoch} 步 = {total_steps} 步")
    print(f"batch     {args.batch_size}")
    print("=" * 68)

    t0 = time.time()
    step = 0
    stopped_early = False

    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        for bi, (planes, target, value) in enumerate(train_loader):
            planes = planes.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            value = value.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                logits, v = net(planes)
                # 稀疏交叉熵：只在非零标签上算，避免 4672 维稠密张量乘法
                lp = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
                lv = F.mse_loss(v.float(), value)
                loss = lp + lv

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            sched.step()

            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                gpu_mem = (torch.cuda.max_memory_allocated() / 1024**3
                           if device.type == "cuda" else 0)
                print(f"ep {epoch}/{args.epochs} step {step:6d} | "
                      f"loss {loss.item():.4f} (p {lp.item():.4f} v {lv.item():.4f}) | "
                      f"lr {sched.get_last_lr()[0]:.2e} | "
                      f"{step/el:.1f} 步/s | "
                      f"显存 {gpu_mem:.1f}G | RSS {rss_mb():.0f}M", flush=True)

            if time.time() - t0 > args.max_hours * 3600:
                print(f"\n[停止] 已跑满 {args.max_hours} 小时，在第 {epoch} epoch 主动收尾")
                stopped_early = True
                break

        # epoch 结束：验证 + 存盘
        vl, vp, vv = evaluate(net, val_loader, device)
        acc = top1_accuracy(net, val_loader, device)
        print(f"  └─ epoch {epoch} val loss {vl:.4f} (p {vp:.4f} v {vv:.4f}) "
              f"· top1 {acc*100:.1f}%", flush=True)

        ckpt = {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "config": {"scale": args.scale},
            "val_loss": vl,
            "top1": acc,
            "epoch": epoch,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.out)                     # 始终保留最新（便于续训）
        if vl < best_val:
            best_val = vl
            torch.save(ckpt, str(args.out).replace(".pt", "-best.pt"))
            print(f"  └─ 最佳模型已保存（{best_val:.4f}）")

        if stopped_early:
            break

    print("=" * 68)
    print(f"完成 · {(time.time()-t0)/60:.1f} 分钟 · best val {best_val:.4f}")
    print(f"输出 {args.out}")
    print("=" * 68)


if __name__ == "__main__":
    main()
