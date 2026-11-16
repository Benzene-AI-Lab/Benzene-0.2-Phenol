#!/usr/bin/env python3
"""
eval_elo.py —— 用 Stockfish 的不同 skill level 估算模型 Elo

原理：
    Stockfish 的 skill level（UCI_LimitStrength + Skill Level）大致对应：
        level 0  ≈ 1350 Elo
        level 3  ≈ 1500
        level 6  ≈ 1700
        level 10 ≈ 1900
        level 15 ≈ 2100
        level 20 ≈ 2400+（接近满强度）

    用模型和这些档位各下 N 局，算胜率：
        score = (win + 0.5*draw) / total
        Elo 差 = -400 * log10(1/score - 1)
        模型 Elo = 对手 Elo + Elo 差

用法：
    python eval_elo.py --ckpt ../weights/0.2-init-alkane-best.pt \\
        --levels 0 3 6 --games 10 --sims 400
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MCTS, pick_move
from net import build as build_net


# Stockfish skill level → 近似 Elo 映射
SKILL_ELO = {0: 1350, 1: 1400, 2: 1450, 3: 1500, 4: 1550, 5: 1600,
             6: 1700, 7: 1750, 8: 1800, 9: 1850, 10: 1900,
             12: 2000, 15: 2100, 18: 2250, 20: 2400}


class StockfishOpponent:
    def __init__(self, path: str, skill: int, depth: int = 1,
                 movetime_ms: int = 0):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.skill = skill
        self.depth = depth
        self.movetime = movetime_ms
        self._send("uci")
        self._wait("uciok")
        # 限制棋力
        self._send("setoption name UCI_LimitStrength value true")
        self._send(f"setoption name Skill Level value {skill}")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 64")
        self._send("isready")
        self._wait("readyok")
        self._send("ucinewgame")

    def _send(self, c: str):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str, timeout: float = 30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.proc.stdout.readline()
            if not line:
                break
            if token in line:
                return
        raise TimeoutError(token)

    def choose(self, board: chess.Board) -> chess.Move:
        self._send(f"position fen {board.fen()}")
        if self.movetime:
            self._send(f"go movetime {self.movetime}")
        else:
            self._send(f"go depth {self.depth}")
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                uci = line.split()[1]
                return chess.Move.from_uci(uci)
            if not line:
                raise RuntimeError("Stockfish 无响应")

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


class NetPlayer:
    def __init__(self, net, sims: int):
        self.mcts = MCTS(net)
        self.sims = sims

    def choose(self, board: chess.Board) -> chess.Move:
        root = self.mcts.run(board, simulations=self.sims)
        _, mv = pick_move(root, temperature=0.0)
        return mv


def play_one(white, black, max_plies: int = 160) -> float:
    """下一局。返回 1 = 白胜, 0.5 = 和棋, 0 = 黑胜。"""
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        player = white if board.turn == chess.WHITE else black
        try:
            mv = player.choose(board)
        except Exception:
            # 出错判负
            return 0.0 if board.turn == chess.WHITE else 1.0
        if mv not in board.legal_moves:
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(mv)

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return 0.5
    return 1.0 if o.winner == chess.WHITE else 0.0


def elo_diff(score: float) -> float:
    """胜率 → Elo 差。score 要夹在 (0,1) 内避免除零。"""
    s = min(max(score, 0.005), 0.995)
    return -400 * math.log10(1 / s - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../weights/0.2-init-alkane-best.pt")
    ap.add_argument("--scale", default="alkane")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6])
    ap.add_argument("--games", type=int, default=10, help="每个 level 下几局")
    ap.add_argument("--sf-depth", type=int, default=1, help="Stockfish 搜索深度")
    ap.add_argument("--sf-movetime", type=int, default=0, help="或按毫秒限时")
    ap.add_argument("--stockfish", default="/tmp/stockfish/stockfish-macos-universal")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--out", default="../runs/elo_report.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck.get("config", {}).get("scale", args.scale))
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"模型 {args.ckpt}  ({net.count_params()/1e6:.2f} M, "
          f"epoch {ck.get('epoch')}, top1 {ck.get('top1',0)*100:.1f}%)")
    print(f"MCTS {args.sims} 次/步  ·  每组 {args.games} 局  ·  上限 {args.max_plies} 步")
    print("=" * 66)

    net_player = NetPlayer(net, args.sims)
    results = []
    t0 = time.time()

    for level in args.levels:
        sf = StockfishOpponent(args.stockfish, level, depth=args.sf_depth,
                               movetime_ms=args.sf_movetime)
        wins = draws = losses = 0
        for g in range(args.games):
            # 交替执白执黑，消除先手优势
            net_white = (g % 2 == 0)
            if net_white:
                score = play_one(net_player, sf, args.max_plies)
            else:
                score = 1.0 - play_one(sf, net_player, args.max_plies)
            if score > 0.75:
                wins += 1
            elif score < 0.25:
                losses += 1
            else:
                draws += 1
            sym = "胜" if score > 0.75 else ("负" if score < 0.25 else "和")
            print(f"  level {level} 第 {g+1}/{args.games} 局（模型执"
                  f"{'白' if net_white else '黑'}）: {sym}  "
                  f"[累计 {wins}/{draws}/{losses}]", flush=True)
        sf.close()

        total = wins + draws + losses
        s = (wins + 0.5 * draws) / max(total, 1)
        ref = SKILL_ELO.get(level, 1350 + level * 55)
        est = ref + elo_diff(s)
        results.append({"level": level, "ref_elo": ref, "wins": wins,
                        "draws": draws, "losses": losses,
                        "score": round(s, 3), "estimated_elo": round(est)})
        print(f"  → level {level}: {wins}胜 {draws}和 {losses}负 "
              f"· 得分率 {s*100:.0f}% · 估计 Elo {est:.0f}")
        print()

    # 汇总
    if results:
        best = max(results, key=lambda r: r["score"])
        print("=" * 66)
        print("汇总")
        for r in results:
            print(f"  level {r['level']:2d} (≈{r['ref_elo']} Elo): "
                  f"{r['wins']}胜 {r['draws']}和 {r['losses']}负 → {r['estimated_elo']} Elo")
        print(f"\n最佳匹配：level {best['level']} · 估计 {best['estimated_elo']} Elo")
        print(f"总耗时 {(time.time()-t0)/60:.1f} 分钟")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ckpt": str(args.ckpt), "sims": args.sims,
                   "results": results}, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"报告 {args.out}")


if __name__ == "__main__":
    main()
