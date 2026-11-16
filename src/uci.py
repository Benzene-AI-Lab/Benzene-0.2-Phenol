#!/usr/bin/env python3
"""
uci.py —— UCI 协议封装

让 Benzene 能被任何支持 UCI 的国际象棋软件使用：
    CuteChess / Arena / Lichess（通过 lichess-bot）/ Fritz / ChessBase

用法：
    python uci.py --ckpt weights/benzene-0.2-phenol.pt --sims 800

在 GUI 里添加引擎时，填：
    Command:  python
    Args:     /path/to/uci.py --ckpt /path/to/weights.pt --sims 800

调试（手动输入 UCI 命令）：
    echo -e "uci\nisready\nposition startpos\ngo movetime 1000\nquit" | \\
        python uci.py --ckpt weights.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MCTS, pick_move, visit_distribution
from net import build as build_net

ENGINE_NAME = "Benzene"
ENGINE_AUTHOR = "CollinsLiu615"


def log(msg: str):
    """UCI 调试信息走 stderr，stdout 只放协议内容。"""
    print(msg, file=sys.stderr, flush=True)


class UCIEngine:
    def __init__(self, ckpt: str, sims: int, device: str = "cpu"):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        scale = ck.get("config", {}).get("scale", "alkane")
        self.net = build_net(scale).to(device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.mcts = MCTS(self.net, device=device)
        self.sims = sims
        self.params = self.net.count_params()
        self.board = chess.Board()
        self.scale = scale
        log(f"[Benzene] 加载 {Path(ckpt).name} · {scale} · {self.params/1e6:.2f}M")

    # ── 命令处理 ──────────────────────────────────────

    def cmd_uci(self):
        out = [
            f"id name {ENGINE_NAME} {self.scale} {self.params/1e6:.2f}M",
            f"id author {ENGINE_AUTHOR}",
            "option name Simulations type spin default 800 min 1 max 100000",
            "option name Temperature type spin default 0 min 0 max 500",
            "uciok",
        ]
        self._send(out)

    def cmd_isready(self):
        self._send(["readyok"])

    def cmd_ucinewgame(self):
        self.board = chess.Board()

    def cmd_setoption(self, parts: list[str]):
        # setoption name <Name> value <Value>
        try:
            i = parts.index("name")
            j = parts.index("value")
            name = " ".join(parts[i + 1:j]).strip().lower()
            value = " ".join(parts[j + 1:]).strip()
        except ValueError:
            return
        if name == "simulations":
            self.sims = int(value)
            log(f"[Benzene] sims = {self.sims}")
        elif name == "temperature":
            pass   # 对局中固定 temperature=0

    def cmd_position(self, parts: list[str]):
        if "startpos" in parts:
            self.board = chess.Board()
            moves_at = parts.index("moves") if "moves" in parts else -1
        elif "fen" in parts:
            i = parts.index("fen")
            j = parts.index("moves") if "moves" in parts else len(parts)
            fen = " ".join(parts[i + 1:j])
            try:
                self.board = chess.Board(fen)
            except ValueError as e:
                log(f"[Benzene] FEN 错误: {e}")
                self.board = chess.Board()
            moves_at = parts.index("moves") if "moves" in parts else -1
        else:
            return

        if moves_at >= 0:
            for uci in parts[moves_at + 1:]:
                try:
                    self.board.push(chess.Move.from_uci(uci))
                except ValueError:
                    log(f"[Benzene] 走法非法: {uci}")
                    break

    def cmd_go(self, parts: list[str]):
        t0 = time.time()
        b = self.board

        if b.is_game_over(claim_draw=True):
            self._send(["bestmove 0000"])
            return

        # 解析时间限制（用不上就不管，固定用 sims）
        nodes = None
        if "nodes" in parts:
            try:
                nodes = int(parts[parts.index("nodes") + 1])
            except (ValueError, IndexError):
                pass

        sims = self.sims
        if nodes:
            sims = min(nodes, 100000)

        root = self.mcts.run(b, simulations=sims)
        _, mv = pick_move(root, temperature=0.0)
        dist = visit_distribution(root)

        # info 行：PV 取访问最多的前几个
        top = sorted(dist.items(), key=lambda x: -x[1])[:5]
        pv = " ".join(root.idx_to_move[i].uci() for i, _ in top if i in root.idx_to_move)
        score_cp = int(max(-10000, min(10000, root.w / max(root.n, 1) * 1000)))
        elapsed = int((time.time() - t0) * 1000)
        n_nodes = sum(ch.n for ch in root.children.values())

        self._send([
            f"info depth 1 seldepth {len(top)} score cp {score_cp} "
            f"nodes {n_nodes} nps {int(n_nodes / max(elapsed/1000, 1e-3))} "
            f"time {elapsed} pv {pv}",
            f"bestmove {mv.uci() if mv else '0000'}",
        ])

    def cmd_stop(self):
        pass   # 引擎是同步的，收到 stop 时已经在算完

    @staticmethod
    def _send(lines):
        for line in lines:
            print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    eng = UCIEngine(args.ckpt, args.sims, args.device)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "uci":
            eng.cmd_uci()
        elif cmd == "isready":
            eng.cmd_isready()
        elif cmd == "ucinewgame":
            eng.cmd_ucinewgame()
        elif cmd == "setoption":
            eng.cmd_setoption(parts)
        elif cmd == "position":
            eng.cmd_position(parts)
        elif cmd == "go":
            eng.cmd_go(parts)
        elif cmd == "stop":
            eng.cmd_stop()
        elif cmd == "quit":
            break
        elif cmd == "ponderhit":
            pass
        else:
            log(f"[Benzene] 未知命令: {line}")

    log("[Benzene] 退出")


if __name__ == "__main__":
    main()
