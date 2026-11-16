#!/usr/bin/env python3
"""
pgn_to_labels.py —— 把 PGN 棋谱转成训练标签

与 gen_labels.py 的区别：
    gen_labels.py     Stockfish 自我对局 → 标签
    pgn_to_labels.py  人类棋谱 → 每个局面跑 Stockfish → 标签

核心逻辑：
    1. 解析 PGN，逐局面遍历
    2. 每个局面让 Stockfish 给出前 3 候选 + 评分
    3. 存成和 labels_100k.npz 完全相同的格式，可以直接混合训练

为什么不能只用人类的走法当标签：
    人类走法是"结果"，不是"评分"。同一个局面，卡帕布兰卡和卡尔森
    可能走不同的棋，都好。只学走法会把模型搞晕。
    有了 Stockfish 评分，模型才能知道"这步棋为什么好"。

用法：
    python pgn_to_labels.py --pgn /tmp/carlsen_pgn/carlsen-magnus.pgn \\
        --out ../data/human_carlsen.npz --max-games 500
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from board import NUM_MOVES
from engine import board_to_planes
from gen_labels import Stockfish, build_policy, cp_to_value
from train import collate  # noqa: F401  (确保模块可导入)


def iter_games(pgn_path: str, max_games: int = 0):
    """逐局读取 PGN。"""
    n = 0
    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            # 过滤：必须有结果、步数合理
            result = game.headers.get("Result", "*")
            # 有些来源（如丁立人集合）没有 Result 头，只要有走法就接受
            moves = list(game.mainline_moves())
            if len(moves) < 10:
                continue
            if result == "*" and moves:
                # 部分来源把结果放在...好吧，无法判断就用和棋占位
                pass
            if result not in ("1-0", "0-1", "1/2-1/2"):
                result = "*"
            n += 1
            if max_games and n > max_games:
                break
            yield game, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="PGN 文件或目录")
    ap.add_argument("--stockfish", default="", help="留空则自动探测")
    ap.add_argument("--out", default="../data/human_labels.npz")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--max-plies", type=int, default=100, help="每局最多取前 N 步")
    ap.add_argument("--min-rating", type=int, default=0, help="只保留双方等级分都高于此值的对局")
    args = ap.parse_args()

    pgn_files = []
    p = Path(args.pgn)
    if p.is_dir():
        pgn_files = sorted(p.glob("*.pgn"))
    else:
        pgn_files = [p]
    if not pgn_files:
        print("找不到 PGN 文件")
        return

    print(f"PGN 文件 {len(pgn_files)} 个")

    sf = Stockfish(args.stockfish, threads=args.threads)

    planes_list: list[np.ndarray] = []
    policy_idx: list[int] = []
    policy_prob: list[float] = []
    policy_ptr: list[int] = [0]
    value_list: list[float] = []
    meta: list[dict] = []

    t0 = time.time()
    total_pos = 0
    games_done = 0
    skipped_rating = 0

    for path in pgn_files:
        print(f"\n处理 {path.name}")
        for game, result in iter_games(str(path), args.max_games):
            # 等级分过滤
            try:
                wr = int(game.headers.get("WhiteElo", "0") or 0)
                br = int(game.headers.get("BlackElo", "0") or 0)
            except ValueError:
                wr = br = 0
            if args.min_rating and (wr < args.min_rating or br < args.min_rating):
                skipped_rating += 1
                continue

            games_done += 1
            board = game.board()
            plies = 0
            for move in game.mainline_moves():
                if plies >= args.max_plies:
                    break
                if board.is_game_over(claim_draw=True):
                    break

                best, cp, cands = sf.analyse(board, depth=args.depth, multipv=3)
                pol = build_policy(board, cands)
                if pol:
                    planes_list.append(board_to_planes(board))
                    for i, pr in pol.items():
                        policy_idx.append(i)
                        policy_prob.append(pr)
                    policy_ptr.append(len(policy_idx))
                    value_list.append(cp_to_value(cp))
                    meta.append({"event": game.headers.get("Event", ""),
                                 "white": game.headers.get("White", ""),
                                 "black": game.headers.get("Black", ""),
                                 "result": result})
                    total_pos += 1

                board.push(move)
                plies += 1

            elapsed = time.time() - t0
            rate = total_pos / max(elapsed, 1e-9)
            if games_done % 10 == 0:
                print(f"  已处理 {games_done} 局  |  {total_pos} 局面  |  "
                      f"{rate:.1f} 局面/秒  |  {elapsed/60:.1f} 分钟", flush=True)

    sf.close()

    if not total_pos:
        print("没有生成任何样本")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        planes=np.stack(planes_list).astype(np.float32),
        policy_idx=np.array(policy_idx, dtype=np.int32),
        policy_prob=np.array(policy_prob, dtype=np.float32),
        policy_ptr=np.array(policy_ptr, dtype=np.int64),
        value=np.array(value_list, dtype=np.float32),
    )

    print("\n" + "=" * 60)
    print(f"完成：{games_done} 局 · {total_pos} 局面")
    if skipped_rating:
        print(f"跳过（等级分不符）：{skipped_rating} 局")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"输出 {args.out}")


if __name__ == "__main__":
    main()
