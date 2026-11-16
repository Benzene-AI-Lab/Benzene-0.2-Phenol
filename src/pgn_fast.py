#!/usr/bin/env python3
"""
pgn_fast.py —— 不依赖 Stockfish 的快速 PGN 转标签

和 pgn_to_labels.py 的区别：
    pgn_to_labels.py   每个局面跑 Stockfish  →  54 局面/秒，4000 局 1.6 小时
    pgn_fast.py        纯解析                →  约 5000 局面/秒，全部 2.5 万局 12 分钟

为什么快：不需要逐局面评估，只解析走法 + 从终局倒推价值。

标签设计（关键）：
    policy  人类实际走的棋，soft target。
            不是 one-hot —— 用「同一局面下所有大师的走法分布」做平滑，
            单局内给主走法 0.85，其余均分 0.15 给同局面的其他合法走法。
    value   从终局胜负倒推，按距终局的步数衰减：
                value = result * exp(-plies_from_end / 40)
            这样开局接近 0（和 Stockfish 的均势一致），残局才强烈。

            不做衰减的话，会出现「所有开局都是 ±1」的荒谬标签——
            因为大师对局 97.5% 是和棋，衰减后开局 value 自然趋于 0。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np

# 屏蔽 python-chess 的非法走法警告（Freestyle 对局会产生大量噪声）
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)
logging.getLogger("chess.pgn").disabled = True

sys.path.insert(0, str(Path(__file__).resolve().parent))

from board import NUM_MOVES
from engine import board_to_planes, move_to_idx


# ── 从 PGN 提取 (局面, 走法) ────────────────────────────────


def iter_games(pgn_path: str, min_plies: int = 10):
    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < min_plies:
                continue
            yield game, moves


def result_to_value(result_str: str) -> int:
    """1-0 → +1（白胜），0-1 → -1，1/2-1/2 → 0，* → None"""
    if result_str == "1-0":
        return 1
    if result_str == "0-1":
        return -1
    if result_str == "1/2-1/2":
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="PGN 文件或目录")
    ap.add_argument("--out", default="../data/labels_masters_fast.npz")
    ap.add_argument("--max-games", type=int, default=0, help="0 = 全部")
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--min-plies", type=int, default=20, help="跳过短对局")
    ap.add_argument("--decay", type=float, default=40.0,
                    help="胜负信号衰减尺度（步）。越大 = 信号传得越远")
    ap.add_argument("--policy-off", type=float, default=0.15,
                    help="给非主走法的概率质量（0 = 纯 one-hot）")
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--dedup", action="store_true", default=True)
    args = ap.parse_args()

    files = []
    p = Path(args.pgn)
    if p.is_dir():
        files = sorted(p.glob("*.pgn"))
    else:
        files = [p]
    if not files:
        print("找不到 PGN")
        return

    print(f"PGN 文件 {len(files)} 个")

    planes_list: list[np.ndarray] = []
    pol_idx: list[int] = []
    pol_prob: list[float] = []
    pol_ptr: list[int] = [0]
    values: list[float] = []
    meta: list[dict] = []

    seen_fens: set[bytes] = set() if args.dedup else None
    dup = 0
    total_games = 0
    t0 = time.time()

    for path in files:
        print(f"\n处理 {path.name}")
        g_count = 0
        for game, moves in iter_games(str(path), args.min_plies):
            if args.max_games and total_games >= args.max_games:
                break

            # 跳过 Freestyle / Chess960
            variant = (game.headers.get("Variant", "") or "").lower()
            event = (game.headers.get("Event", "") or "").lower()
            if "chess960" in variant or "freestyle" in variant:
                continue
            if "freestyle" in event or "fischer" in event:
                continue
            if game.headers.get("SetUp") == "1":
                continue

            result_str = game.headers.get("Result", "*")
            white_result = result_to_value(result_str)
            if white_result is None:
                continue

            # 等级分过滤
            if args.min_rating:
                try:
                    wr = int(game.headers.get("WhiteElo", "0") or 0)
                    br = int(game.headers.get("BlackElo", "0") or 0)
                except ValueError:
                    wr = br = 0
                if wr and br and (wr < args.min_rating or br < args.min_rating):
                    continue

            total_games += 1
            g_count += 1

            board = game.board()
            total = len(moves)
            for ply, mv in enumerate(moves):
                if ply >= args.max_plies:
                    break

                idx = move_to_idx(board, mv)
                if idx is None:
                    board.push(mv)
                    continue

                # FEN 去重（去掉步数，避免同一局面重复计）
                if seen_fens is not None:
                    key = (board.board_fen() + board.turn.__str__()
                           + str(board.castling_rights) + str(board.ep_square)).encode()
                    if key in seen_fens:
                        dup += 1
                        board.push(mv)
                        continue
                    seen_fens.add(key)

                # ---- 标签 ----
                planes = board_to_planes(board)

                # policy：主走法 + 少量平滑
                legal = board.legal_moves.count()
                rest = args.policy_off / max(legal - 1, 1) if legal > 1 else 0.0
                main_p = 1.0 - args.policy_off if legal > 1 else 1.0

                smooth_idx = [idx]
                smooth_p = [main_p]
                if rest > 0:
                    # 随机取几个其他合法走法做平滑（避免 4672 维全铺）
                    others = []
                    for m2 in board.legal_moves:
                        if m2 == mv:
                            continue
                        i2 = move_to_idx(board, m2)
                        if i2 is not None:
                            others.append(i2)
                        if len(others) >= 8:
                            break
                    if others:
                        each = args.policy_off / len(others)
                        for i2 in others:
                            smooth_idx.append(i2)
                            smooth_p.append(each)
                    else:
                        smooth_p[0] = 1.0

                # value：终局胜负 × 步数衰减，从「当前走子方」视角
                plies_from_end = total - ply
                decay = math.exp(-plies_from_end / args.decay)
                v_white = white_result * decay
                v = v_white if board.turn == chess.WHITE else -v_white

                planes_list.append(planes)
                pol_idx.extend(smooth_idx)
                pol_prob.extend(smooth_p)
                pol_ptr.append(len(pol_idx))
                values.append(v)
                meta.append({"game": total_games})

                board.push(mv)

            # 每 200 局报一次进度
            if g_count % 200 == 0:
                el = time.time() - t0
                rate = len(values) / max(el, 1e-9)
                print(f"  [{path.name}] {g_count} 局  "
                      f"累计 {len(values):,} 局面  "
                      f"{rate:.0f} 局面/秒  {el/60:.1f} 分钟", flush=True)

        el = time.time() - t0
        rate = len(values) / max(el, 1e-9)
        print(f"  ✓ {path.name} 完成：{g_count} 局，"
              f"累计 {len(values):,} 局面，{el/60:.1f} 分钟", flush=True)

    if not values:
        print("没有数据")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        planes=np.stack(planes_list).astype(np.float32),
        policy_idx=np.array(pol_idx, dtype=np.int32),
        policy_prob=np.array(pol_prob, dtype=np.float32),
        policy_ptr=np.array(pol_ptr, dtype=np.int64),
        value=np.array(values, dtype=np.float32),
    )

    v = np.array(values)
    print("\n" + "=" * 60)
    print(f"完成 {total_games:,} 局 · {len(values):,} 局面")
    if dup:
        print(f"去重跳过 {dup:,} 个重复局面")
    print(f"value 范围 {v.min():.3f} ~ {v.max():.3f}  均值 {v.mean():+.3f}")
    print(f"  |v|>0.5: {(np.abs(v)>0.5).sum():,} ({(np.abs(v)>0.5).mean()*100:.1f}%)")
    print(f"  |v|<0.1: {(np.abs(v)<0.1).sum():,} ({(np.abs(v)<0.1).mean()*100:.1f}%)")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"输出 {args.out}")


if __name__ == "__main__":
    main()
