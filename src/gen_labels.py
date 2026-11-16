#!/usr/bin/env python3
"""
gen_labels.py —— 用 Stockfish 生成监督学习标签

输出每个局面：
    planes   19×8×8 特征
    policy   4672 维稀疏分布（前 N 个候选 + 概率）
    value    归一化评分 [-1, 1]

价值归一化：
    cp  = clamp(stockfish_cp, -1500, 1500)
    value = tanh(cp / 400)
    400cp ≈ 一个兵的劣势，tanh 让极端分数饱和，避免异常值主导

depth 选择：
    6  → 60 局面/秒，标签略粗糙
    8  → 86 局面/秒，性价比最高  ← 默认
    10 → 12 局面/秒，最准但慢 8 倍

用法：
    python gen_labels.py --games 200 --depth 8 --out ../data/labels_200k.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from board import NUM_MOVES
from engine import board_to_planes, legal_move_indices, move_to_idx


def find_stockfish(explicit: str = "") -> str:
    """按优先级找可用的 Stockfish 可执行文件。"""
    import glob
    candidates = [
        explicit,
        "/tmp/stockfish/stockfish-ubuntu-x86-64-avx2",
        "/tmp/stockfish/stockfish-ubuntu-x86-64",
        "/tmp/stockfish/stockfish-macos-universal",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # 最后兜底：在 /tmp/stockfish 下任意找
    for c in glob.glob("/tmp/stockfish/stockfish*"):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "找不到 Stockfish，请下载 Linux 版到 /tmp/stockfish/：\n"
        "  curl -sL -o /tmp/sf.tar.gz https://github.com/official-stockfish/"
        "Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz\n"
        "  tar xzf /tmp/sf.tar.gz -C /tmp/stockfish --strip-components=1\n"
        "  chmod +x /tmp/stockfish/*"
    )


class Stockfish:
    def __init__(self, path: str, threads: int = 2, hash_mb: int = 256,
                 multipv: int = 1):
        path = find_stockfish(path)
        self.path = path
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name MultiPV value {multipv}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.proc.stdout.readline()
            if not line:
                break
            if token in line:
                return line
        raise TimeoutError(f"等待 {token} 超时")

    def analyse(self, board: chess.Board, depth: int, multipv: int = 3):
        """返回 (best_move_uci, best_cp, [(move_uci, cp), ...])。"""
        self._send(f"position fen {board.fen()}")
        self._send(f"go depth {depth}")

        lines: dict[int, tuple[str, int]] = {}    # multipv 序号 → (move, cp)
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                continue
            if line.startswith("bestmove"):
                best = line.split()[1]
                break
            if not line.startswith("info") or " pv " not in line:
                continue
            try:
                parts = line.split()
                mpv = 1
                if "multipv" in parts:
                    mpv = int(parts[parts.index("multipv") + 1])
                if " mate " in line:
                    mate = int(line.split(" mate ")[1].split()[0])
                    cp = 30000 if mate > 0 else -30000
                elif " cp " in line:
                    cp = int(line.split(" cp ")[1].split()[0])
                else:
                    continue
                pv_i = parts.index("pv")
                mv = parts[pv_i + 1]
                lines[mpv] = (mv, cp)
            except (ValueError, IndexError):
                continue

        if not lines:
            return None, 0, []
        cands = [lines[k] for k in sorted(lines)]
        best_cp = cands[0][1]
        return best, best_cp, cands

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def cp_to_value(cp: int) -> float:
    cp = max(-1500, min(1500, cp))
    return float(np.tanh(cp / 400.0))


def build_policy(board: chess.Board, cands: list[tuple[str, int]],
                 top_n: int = 3) -> dict[int, float]:
    """把 Stockfish 的候选走法转成 {index: prob}。

    概率分配：用 cp 差距做 softmax，温度 100cp。
    """
    scored = []
    for uci, cp in cands[:top_n]:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if mv not in board.legal_moves:
            continue
        idx = move_to_idx(board, mv)
        if idx is None:
            continue
        scored.append((idx, cp))

    if not scored:
        return {}

    base = scored[0][1]
    weights = [np.exp((cp - base) / 100.0) for _, cp in scored]
    total = sum(weights)
    return {i: w / total for (i, _), w in zip(scored, weights)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", default="", help="留空则自动探测")
    ap.add_argument("--games", type=int, default=50, help="对局数（每局约 40-80 局面）")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--multipv", type=int, default=1,
                    help="候选走法数。1 最快（约 340 局面/秒），3 慢 19 倍")
    ap.add_argument("--out", default="../data/labels.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--min-ply", type=int, default=0,
                    help="跳过开局前 N 步（用于补中后局数据）")
    ap.add_argument("--tail-weight", type=float, default=1.0,
                    help="后期局面的采样权重倍数（>1 偏向中后局）")
    ap.add_argument("--openings", default="", help="可选：开局 UCI 列表文件，每行一串走法")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    sf = Stockfish(args.stockfish, threads=args.threads, multipv=args.multipv)

    # 开局集合：默认从起始局面 + 随机偏离
    openings = []
    if args.openings and os.path.exists(args.openings):
        for line in open(args.openings, encoding="utf-8"):
            line = line.strip()
            if line:
                openings.append(line)

    planes_list: list[np.ndarray] = []
    policy_idx: list[int] = []
    policy_prob: list[float] = []
    policy_ptr: list[int] = [0]
    value_list: list[float] = []
    meta: list[dict] = []

    t_start = time.time()
    total_pos = 0

    for g in range(args.games):
        sf._send('ucinewgame')
        board = chess.Board()
        # 开局随机化：前 4 步用 Stockfish 的前 3 候选随机选，保证多样性
        n_open = random.randint(0, 3)
        for _ in range(n_open):
            if board.is_game_over(claim_draw=True):
                break
            _, _, cands = sf.analyse(board, depth=args.depth, multipv=args.multipv)
            if not cands:
                break
            uci = random.choice([c[0] for c in cands])
            try:
                board.push(chess.Move.from_uci(uci))
            except ValueError:
                break

        ply = 0
        while ply < args.max_plies:
            if board.is_game_over(claim_draw=True):
                break

            best, cp, cands = sf.analyse(board, depth=args.depth, multipv=args.multipv)
            if not cands:
                break

            pol = build_policy(board, cands)
            # 采样：跳过开局 / 加权后期局面
            take = pol and ply >= args.min_ply
            if take and args.tail_weight > 1.0:
                # ply 越靠后，保留概率越高
                frac = ply / max(args.max_plies, 1)
                keep_p = min(1.0, frac * args.tail_weight)
                if random.random() > keep_p:
                    take = False
            if take:
                planes_list.append(board_to_planes(board))
                for i, p in pol.items():
                    policy_idx.append(i)
                    policy_prob.append(p)
                policy_ptr.append(len(policy_idx))
                value_list.append(cp_to_value(cp))
                total_pos += 1

            # 走一步：前 10 步随机选（探索），之后走最优
            if ply < 10:
                uci = random.choice([c[0] for c in cands])
            else:
                uci = best
            try:
                board.push(chess.Move.from_uci(uci))
            except ValueError:
                break
            ply += 1

        elapsed = time.time() - t_start
        rate = total_pos / max(elapsed, 1e-9)
        print(f"  局 {g+1}/{args.games}  {ply} 步  |  累计 {total_pos} 局面  "
              f"|  {rate:.1f} 局面/秒  |  {elapsed/60:.1f} 分钟", flush=True)

    sf.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        planes=np.stack(planes_list).astype(np.float32) if planes_list else np.zeros((0, 19, 8, 8), np.float32),
        policy_idx=np.array(policy_idx, dtype=np.int32),
        policy_prob=np.array(policy_prob, dtype=np.float32),
        policy_ptr=np.array(policy_ptr, dtype=np.int64),
        value=np.array(value_list, dtype=np.float32),
    )
    print(f"\n完成：{total_pos} 个局面 → {args.out}")
    print(f"总耗时 {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
