#!/usr/bin/env python3
"""
play.py —— 最小可跑的对弈与自我对弈

两个用途：
    1. 看模型下棋（人类可读的棋谱）
    2. 生成自我对弈数据（供训练）

本机 i5 上，一次 MCTS 模拟约 29 ms：
    每步 100 次模拟 → 2.9 秒
    一局约 80 步   → 4 分钟
CPU 云机（8 核）应快 3-4 倍。
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Optional

import chess
import numpy as np
import torch

from engine import (MCTS, board_to_planes, legal_move_indices, pick_move,
                    visit_distribution)
from net import ChessNet


# ── 随机对手 ────────────────────────────────────────────────


class RandomPlayer:
    name = "random"

    def choose(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class NetPlayer:
    name = "net"

    def __init__(self, net: ChessNet, simulations: int = 100,
                 temperature: float = 0.0, device: str = "cpu"):
        self.mcts = MCTS(net, device=device)
        self.simulations = simulations
        self.temperature = temperature

    def choose(self, board: chess.Board) -> chess.Move:
        root = self.mcts.run(board, simulations=self.simulations)
        _, mv = pick_move(root, temperature=self.temperature)
        return mv


# ── 单局对弈 ────────────────────────────────────────────────


def play_game(white, black, max_moves: int = 200,
              record: bool = False, verbose: bool = True) -> dict:
    """下一局。返回 {result, moves, plies, samples}。

    result 从白方视角：1 = 白胜，-1 = 黑胜，0 = 和棋。
    """
    board = chess.Board()
    moves: list[str] = []
    samples: list[dict] = []

    for ply in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break

        player = white if board.turn == chess.WHITE else black

        # 记录训练样本：局面 + 最终胜负（胜负后面回填）
        if record:
            planes = board_to_planes(board)
            legal = legal_move_indices(board)
        else:
            planes = legal = None

        t0 = time.time()
        move = player.choose(board)
        dt = time.time() - t0

        if record:
            # policy 标签：用 player 的搜索分布（若支持），否则用 one-hot
            if isinstance(player, NetPlayer):
                # 这里记录 one-hot；带 visit 分布的版本见 selfplay_game()
                target = {i: 0.0 for i in legal}
                mi = None
                for i, mv in legal.items():
                    if mv == move:
                        mi = i
                        break
                if mi is not None:
                    target[mi] = 1.0
                samples.append({"planes": planes, "policy": target,
                                "turn": bool(board.turn)})
            else:
                samples.append({"planes": planes, "policy": None,
                                "turn": bool(board.turn)})

        moves.append(move.uci())
        board.push(move)

        if verbose and ply % 10 == 0:
            print(f"    ply {ply:3d}  {move.uci():6s}  ({dt*1000:.0f} ms)")

    # 结果
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result = 0                  # 超过步数上限，算和棋
    elif outcome.winner is None:
        result = 0
    else:
        result = 1 if outcome.winner == chess.WHITE else -1

    # 回填 value
    for s in samples:
        # 从该局面的走子方视角
        s["value"] = result if s["turn"] else -result

    if verbose:
        reason = outcome.termination.name if outcome else "MAX_MOVES"
        print(f"    结果: {result:+d}  ({reason})  共 {len(moves)} 步")

    return {"result": result, "moves": moves, "plies": len(moves),
            "samples": samples}


# ── 自我对弈（带 visit 分布）────────────────────────────────


def selfplay_game(net: ChessNet, simulations: int = 100,
                  temperature: float = 1.0, temp_drop_ply: int = 30,
                  device: str = "cpu", verbose: bool = False) -> dict:
    """自我对弈一局，返回带 MCTS visit 分布的样本。

    temperature 在前 temp_drop_ply 步用 1.0（探索），之后降到 0.25（收敛）。
    """
    mcts = MCTS(net, device=device)
    board = chess.Board()
    samples = []
    moves = []

    for ply in range(300):
        if board.is_game_over(claim_draw=True):
            break

        root = mcts.run(board, simulations=simulations, add_noise=True)
        planes = board_to_planes(board)
        dist = visit_distribution(root)

        temp = temperature if ply < temp_drop_ply else 0.25
        idx, move = pick_move(root, temperature=temp)

        samples.append({"planes": planes, "policy": dist, "turn": bool(board.turn)})
        moves.append(move.uci())
        board.push(move)

        if verbose and ply % 10 == 0:
            print(f"    ply {ply:3d}  {move.uci()}")

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result = 0
    else:
        result = 1 if outcome.winner == chess.WHITE else -1

    for s in samples:
        s["value"] = result if s["turn"] else -result

    return {"result": result, "moves": moves, "samples": samples}


# ── 主程序 ──────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Benzene 0.2 · 对弈测试")
    ap.add_argument("--sims", type=int, default=50, help="每步 MCTS 模拟次数")
    ap.add_argument("--opponent", choices=["random", "self"], default="random")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net = ChessNet()
    net.eval()
    print(f"网络: {net.count_params()/1e6:.2f} M  ·  模拟 {args.sims} 次/步")
    print(f"对手: {args.opponent}  ·  局数 {args.games}")
    print("=" * 64)

    wins = draws = losses = 0
    t_start = time.time()

    for g in range(args.games):
        print(f"\n第 {g+1} 局")
        if args.opponent == "random":
            # 模型执白
            white = NetPlayer(net, simulations=args.sims, temperature=0.0,
                              device=args.device)
            black = RandomPlayer()
            r = play_game(white, black, verbose=True)
            if r["result"] > 0:
                wins += 1
            elif r["result"] < 0:
                losses += 1
            else:
                draws += 1
        else:
            r = selfplay_game(net, simulations=args.sims, device=args.device,
                              verbose=True)
            print(f"    结果 {r['result']:+d}")

    dt = time.time() - t_start
    print("=" * 64)
    print(f"耗时 {dt:.1f}s  ({dt/max(args.games,1)/60:.1f} 分钟/局)")
    if args.opponent == "random":
        print(f"战绩: 胜 {wins}  和 {draws}  负 {losses}")


if __name__ == "__main__":
    main()
