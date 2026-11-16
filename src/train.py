#!/usr/bin/env python3
"""
train.py —— 监督学习训练（Init 阶段）

目标：
    policy  预测 Stockfish 的最优走法分布   loss = 交叉熵
    value   预测局面评分归一化值            loss = MSE

用法：
    python train.py --data ../data/labels_200k.npz --epochs 8 --out ../weights/0.2-init.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import NUM_MOVES
from net import ChessNet


class LabelDataset(torch.utils.data.Dataset):
    """把稀疏的 policy 标签展开成稠密 4672 维。

    内存考量：展开后每个局面 4672 float32 = 18.7 KB，
    20 万局面 = 3.7 GB，太占内存。
    所以保持稀疏，在 __getitem__ 里现做。
    """

    def __init__(self, path, val_ratio: float = 0.05, seed: int = 0):
        """path 可以是单文件 npz，也可以是 build_dataset.py 产出的目录。"""
        p = Path(path)

        if p.is_dir() and (p / "planes.npy").exists():
            # 二进制数据集：memmap 载入，不占内存
            def npy(name):
                return np.load(str(p / name), mmap_mode="r")
            self.planes = npy("planes.npy")
            self.policy_idx = npy("policy_idx.npy")
            self.policy_prob = npy("policy_prob.npy")
            self.policy_ptr = npy("policy_ptr.npy")
            self.value = npy("value.npy")
            meta = json.loads((p / "index.json").read_text(encoding="utf-8"))
            srcs = meta.get("sources", [])
        else:
            # 单个 npz
            with np.load(str(p)) as d:
                self.planes = d["planes"]
                self.policy_idx = d["policy_idx"]
                self.policy_prob = d["policy_prob"]
                self.policy_ptr = d["policy_ptr"]
                self.value = d["value"]
            srcs = []

        n = len(self.value)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        self.val_idx = perm[:n_val]
        self.train_idx = perm[n_val:]

        print(f"[数据] {n:,} 个局面  ·  训练 {len(self.train_idx):,}  ·  验证 {len(self.val_idx):,}")
        for s_ in srcs:
            print(f"       {s_['name']}: {s_['positions']:,}")

    def subset(self, idx: np.ndarray) -> "LabelDataset":
        obj = object.__new__(LabelDataset)
        obj.planes = self.planes
        obj.policy_idx = self.policy_idx
        obj.policy_prob = self.policy_prob
        obj.policy_ptr = self.policy_ptr
        obj.value = self.value
        obj.train_idx = idx
        obj.val_idx = idx
        return obj

    def __len__(self) -> int:
        return len(self.train_idx)

    def __getitem__(self, i: int):
        """返回 (planes, idx_slice, prob_slice, value)，稠密化交给 collate。

        每个样本只返回 numpy 视图，不做零张量分配——
        这是 CPU 训练的主要瓶颈（每样本 4,672 float = 18.7 KB）。
        """
        j = self.train_idx[i]
        a, b = int(self.policy_ptr[j]), int(self.policy_ptr[j + 1])
        return (np.asarray(self.planes[j], dtype=np.float32),
                self.policy_idx[a:b].astype(np.int64),
                self.policy_prob[a:b].astype(np.float32),
                np.float32(self.value[j]))


def collate(batch):
    """把 batch 拼成稠密 policy 张量，只分配一次内存。"""
    planes = torch.from_numpy(np.stack([b[0] for b in batch]))
    B = len(batch)
    target = torch.zeros(B, NUM_MOVES, dtype=torch.float32)
    for row, (_, idxs, probs, _) in enumerate(batch):
        target[row, torch.from_numpy(idxs)] = torch.from_numpy(probs)
    value = torch.tensor([[b[3]] for b in batch], dtype=torch.float32)
    return planes, target, value


def evaluate(net: ChessNet, loader, device) -> tuple[float, float, float]:
    net.eval()
    total_loss = total_p = total_v = 0.0
    n = 0
    with torch.inference_mode():
        for planes, target, value in loader:
            planes = planes.to(device)
            target = target.to(device)
            value = value.to(device)
            logits, v = net(planes)
            lp = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
            lv = F.mse_loss(v, value)
            total_loss += (lp + lv).item() * planes.size(0)
            total_p += lp.item() * planes.size(0)
            total_v += lv.item() * planes.size(0)
            n += planes.size(0)
    net.train()
    return total_loss / max(n, 1), total_p / max(n, 1), total_v / max(n, 1)


def top1_accuracy(net: ChessNet, loader, device) -> float:
    """policy 的 top-1 命中率：预测的最高概率走法是否在标签的前 3 候选里。"""
    net.eval()
    hit = tot = 0
    with torch.inference_mode():
        for planes, target, _ in loader:
            planes = planes.to(device)
            target = target.to(device)      # 必须和 logits 同设备
            logits, _ = net(planes)
            pred = logits.argmax(1)
            # 标签把概率分给多个候选，所以检查预测落在非零位置
            hit += int((target.gather(1, pred.unsqueeze(1)).squeeze(1) > 0.01).sum())
            tot += planes.size(0)
    net.train()
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scale", default="alkane")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="../weights/0.2-init.pt")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ds = LabelDataset(args.data)
    train_ds = ds.subset(ds.train_idx)
    val_ds = ds.subset(ds.val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.workers,
    )

    from net import build as build_net
    net = build_net(args.scale)
    net.to(device)
    n_param = net.count_params()

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # 前 3% 步数做 warmup，避免一开始就被大梯度带偏
    warmup = max(1, int(total_steps * 0.03))
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        import math
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print("=" * 64)
    print(f"设备     {device}")
    print(f"参数     {n_param/1e6:.2f} M")
    print(f"训练     {len(train_ds)} 局面 × {args.epochs} epoch = {total_steps} 步")
    print("=" * 64)

    best_loss = float("inf")
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        net.train()
        for planes, target, value in train_loader:
            planes = planes.to(device)
            target = target.to(device)
            value = value.to(device)

            logits, v = net(planes)
            lp = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
            lv = F.mse_loss(v, value)
            loss = lp + lv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()

            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"ep {epoch} step {step:5d}/{total_steps} | "
                      f"loss {loss.item():.4f} (p {lp.item():.4f} v {lv.item():.4f}) | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {step/el:.2f} 步/s", flush=True)

        # 每个 epoch 结束验证
        vl, vp, vv = evaluate(net, val_loader, device)
        acc = top1_accuracy(net, val_loader, device)
        print(f"  └─ val loss {vl:.4f} (p {vp:.4f} v {vv:.4f}) · top1 {acc*100:.1f}%")

        if vl < best_loss:
            best_loss = vl
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": net.state_dict(),
                "config": {"blocks": net.blocks, "channels": net.channels},
                "val_loss": vl,
                "top1": acc,
                "epoch": epoch,
            }, args.out)
            print(f"  └─ 已保存 {args.out}")

    print("=" * 64)
    print(f"完成 · {args.epochs} epoch · {(time.time()-t0)/60:.1f} 分钟 · best val {best_loss:.4f}")


if __name__ == "__main__":
    main()
