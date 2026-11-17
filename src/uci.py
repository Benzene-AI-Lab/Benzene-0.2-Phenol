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

ENGINE_NAME = "Benzene 0.2-Phenol"
ENGINE_AUTHOR = "CollinsLiu615"
ENGINE_VERSION = "1.0"


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
        self.hash_mb = 128
        self.move_overhead_ms = 30
        self.board = chess.Board()
        self.scale = scale
        log(f"[Benzene] 加载 {Path(ckpt).name} · {scale} · {self.params/1e6:.2f}M")

    # ── 命令处理 ──────────────────────────────────────

    def cmd_uci(self):
        out = [
            f"id name {ENGINE_NAME}",
            f"id author {ENGINE_AUTHOR}",
            f"id version {ENGINE_VERSION}",
            "option name Simulations type spin default 800 min 1 max 100000",
            "option name Temperature type spin default 0 min 0 max 500",
            # 标准选项：声明出来以便通过 CCRL / CuteChess / Arena 的兼容检查。
            # Benzene 是固定结构的神经网络，不使用置换表也不多线程，这些值只记录不生效。
            "option name Hash type spin default 128 min 1 max 4096",
            "option name Threads type spin default 1 min 1 max 1",
            "option name Ponder type check default false",
            "option name UCI_Chess960 type check default false",
            "option name UCI_LimitStrength type check default false",
            "option name Move Overhead type spin default 30 min 0 max 5000",
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
        elif name == "hash":
            self.hash_mb = int(value) if value.isdigit() else 128
            log(f"[Benzene] hash = {self.hash_mb} MB（不使用，仅记录）")
        elif name == "threads":
            log(f"[Benzene] threads = {value}（单线程推理）")
        elif name == "move overhead":
            self.move_overhead_ms = int(value) if value.isdigit() else 30
            log(f"[Benzene] move overhead = {self.move_overhead_ms} ms")
        elif name == "ponder":
            pass
        elif name in ("uci_chess960", "uci_limitstrength"):
            pass

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


    # 单次网络前向的耗时（秒）。用一次干跑实测，避免写死。
    _move_s = None

    def _forward_seconds(self) -> float:
        """每次 MCTS 模拟的秒数。实测后加 25% 保险，避免估低导致超时。"""
        if UCIEngine._move_s is None:
            import time as _t
            b = chess.Board()
            t0 = _t.time()
            self.mcts.run(b, simulations=8)
            UCIEngine._move_s = max((_t.time() - t0) / 9.0, 0.005) * 1.8
        return UCIEngine._move_s

    def _sims_for_ms(self, ms: float) -> int:
        """把毫秒预算换算成 MCTS 模拟次数，留 25% 余量。"""
        usable = max(ms - self.move_overhead_ms, 1) / 1000.0 * 0.55
        return max(int(usable / self._forward_seconds()), 8)

    @staticmethod
    def _budget_ms(remaining_ms: int, inc_ms: int, movestogo: int | None) -> float:
        """按剩余时间分配这一手的用时。"""
        if movestogo and movestogo > 0:
            share = remaining_ms / (movestogo + 1)
        else:
            # 无 movestogo 时按「还剩约 30 手」估算，取 1/30 再加增量
            share = remaining_ms / 30.0
        budget = share + inc_ms * 0.8
        # 不要一次用掉超过剩余时间的 1/5，也不要超过 30 秒
        return max(min(budget, remaining_ms * 0.2, 30000), 20)

    def cmd_go(self, parts: list[str]):
        t0 = time.time()
        b = self.board

        if b.is_game_over(claim_draw=True):
            self._send(["bestmove 0000"])
            return

        def arg(name):
            try:
                return int(parts[parts.index(name) + 1])
            except (ValueError, IndexError):
                return None

        # 优先用 go 命令显式给的限制，其次是时钟，最后回退到固定的 sims
        nodes = arg("nodes")
        movetime = arg("movetime")
        wtime, btime = arg("wtime"), arg("btime")
        winc, binc = arg("winc"), arg("binc")
        movestogo = arg("movestogo")

        sims = None
        if nodes:
            sims = min(nodes, 100000)
        elif movetime is not None:
            sims = self._sims_for_ms(movetime)
        elif wtime is not None or btime is not None:
            mine = wtime if b.turn == chess.WHITE else btime
            inc = (winc if b.turn == chess.WHITE else binc) or 0
            if mine is not None:
                sims = self._sims_for_ms(self._budget_ms(mine, inc, movestogo))

        if sims is None:
            sims = self.sims

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
    ap.add_argument("--name", default="",
                    help="覆盖 UCI id name（默认按权重文件名推断）")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    # 按权重文件名推断引擎名：benzene-0.2-phenol.pt -> Benzene 0.2-Phenol
    global ENGINE_NAME
    if args.name:
        ENGINE_NAME = args.name
    else:
        stem = Path(args.ckpt).stem
        parts = stem.split("-")
        if len(parts) >= 3 and parts[0].lower() == "benzene":
            ENGINE_NAME = f"Benzene {parts[1]}-{parts[2].capitalize()}"

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
