#!/usr/bin/env python3
"""
engine.py —— 局面封装 + MCTS 搜索

为什么用 python-chess 做规则：
    王车易位的条件、吃过路兵的时机、升变的四种选择、逼和与三次重复，
    自己实现几乎必然出 bug，而且 bug 会以「模型学出诡异走法」的形式隐蔽地出现。
    python-chess 只做合法性判断，不做任何搜索——棋力全部来自我们的网络。

MCTS：
    标准 PUCT。
        U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        score  = Q(s,a) + U(s,a)
    Q 从当前走子方视角存储，展开时取负（零和）。
"""

from __future__ import annotations

import math
import random
from typing import Optional

import chess
import numpy as np
import torch

from board import NUM_MOVES, NUM_PLANES
from net import ChessNet


# ── 局面 → 张量 ─────────────────────────────────────────────


def board_to_planes(board: chess.Board) -> np.ndarray:
    """chess.Board → (19, 8, 8) float32，以「当前走子方」为第一视角。"""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    me = board.turn                       # True = 白

    for sq, piece in board.piece_map().items():
        # python-chess: a1=0, h8=63；我们的编码: a8=0, h1=63
        r = 7 - (sq // 8)
        c = sq % 8
        is_mine = (piece.color == me)
        idx = piece.piece_type - 1        # PAWN=1 → 0
        plane = idx if is_mine else idx + 6
        planes[plane, r, c] = 1.0

    # 易位权
    if board.has_kingside_castling_rights(me):
        planes[12].fill(1.0)
    if board.has_queenside_castling_rights(me):
        planes[13].fill(1.0)
    opp = not me
    if board.has_kingside_castling_rights(opp):
        planes[14].fill(1.0)
    if board.has_queenside_castling_rights(opp):
        planes[15].fill(1.0)

    # 吃过路兵
    ep = board.ep_square
    if ep is not None:
        r = 7 - (ep // 8)
        c = ep % 8
        planes[16, r, c] = 1.0

    # 轮到谁走
    planes[17].fill(1.0)

    # 步数
    planes[18].fill(min(board.fullmove_number / 100.0, 1.0))

    return planes


# ── 走法 ↔ 索引 ─────────────────────────────────────────────


def move_to_idx(board: chess.Board, move: chess.Move) -> Optional[int]:
    """把 chess.Move 映射到 0..4671。无法表达时返回 None。"""
    from board import move_to_index, square_from_name

    def sq(s: int) -> int:
        # python-chess a1=0 → 我们 a8=0
        r = 7 - (s // 8)
        c = s % 8
        return r * 8 + c

    f, t = sq(move.from_square), sq(move.to_square)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None

    try:
        if piece.piece_type == chess.KNIGHT:
            return move_to_index(f, t, is_knight=True)
        if piece.piece_type == chess.PAWN:
            promo = None
            if move.promotion is not None:
                # 升变：chess.KNIGHT=2 → 我们的 KNIGHT=1
                promo = move.promotion - 1
                # 我们的兵升变只编码斜吃，直进升变不支持（极罕见）
            return move_to_index(f, t, is_pawn=True, promo=promo)
        return move_to_index(f, t)
    except ValueError:
        return None


def legal_move_indices(board: chess.Board) -> dict[int, chess.Move]:
    """返回 {index: chess.Move}，只含网络能表达的走法。"""
    out = {}
    for mv in board.legal_moves:
        i = move_to_idx(board, mv)
        if i is not None and i not in out:
            out[i] = mv
    return out


# ── MCTS ────────────────────────────────────────────────────


class Node:
    __slots__ = ("prior", "n", "w", "children", "expanded", "idx_to_move")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.children: dict[int, "Node"] = {}
        self.expanded = False
        self.idx_to_move: dict[int, chess.Move] = {}

    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


class MCTS:
    def __init__(self, net: ChessNet, c_puct: float = 1.5,
                 device: str = "cpu", dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25):
        self.net = net
        self.c_puct = c_puct
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.net.eval()

    # ---- 网络评估 ----
    @torch.inference_mode()
    def _evaluate(self, board: chess.Board) -> tuple[dict[int, float], float]:
        """返回 ({move_idx: prob}, value)。value 是当前走子方视角 [-1,1]。"""
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, v = self.net(x)
        logits = logits[0]
        probs = torch.softmax(logits, dim=0)

        legal = legal_move_indices(board)
        if not legal:
            return {}, float(v.item())

        idxs = list(legal.keys())
        priors = probs[torch.tensor(idxs, device=self.device)].cpu().numpy()
        s = priors.sum()
        if s > 0:
            priors = priors / s
        else:
            priors = np.ones(len(idxs), dtype=np.float32) / len(idxs)

        return {i: float(p) for i, p in zip(idxs, priors)}, float(v.item())

    # ---- 单次模拟 ----
    def _simulate(self, board: chess.Board, root: Node, add_noise: bool) -> float:
        path: list[Node] = [root]
        node = root

        while True:
            if board.is_game_over(claim_draw=True):
                return _terminal_value(board)

            if not node.expanded:
                priors, value = self._evaluate(board)
                if not priors:
                    return _terminal_value(board)

                if add_noise and node is root:
                    priors = self._add_dirichlet(priors)

                legal = legal_move_indices(board)
                for i, p in priors.items():
                    node.children[i] = Node(p)
                node.idx_to_move = legal
                node.expanded = True
                return value

            # 选子节点
            idx = self._select(node)
            move = node.idx_to_move[idx]
            board.push(move)
            node = node.children[idx]
            path.append(node)

    def _select(self, node: Node) -> int:
        sqrtN = math.sqrt(max(node.n, 1))
        best, best_score = -1, -1e18
        for i, ch in node.children.items():
            u = self.c_puct * ch.prior * sqrtN / (1 + ch.n)
            # Q 从父节点（对手）视角：子节点 value 是子方视角，取负
            score = -ch.q() + u
            if score > best_score:
                best_score, best = score, i
        return best

    def _add_dirichlet(self, priors: dict[int, float]) -> dict[int, float]:
        n = len(priors)
        if n == 0:
            return priors
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        keys = list(priors.keys())
        eps = self.dirichlet_eps
        return {k: (1 - eps) * priors[k] + eps * float(noise[j])
                for j, k in enumerate(keys)}

    # ---- 对外接口 ----
    def run(self, board: chess.Board, simulations: int = 200,
            add_noise: bool = False) -> Node:
        """跑 MCTS，返回根节点。调用后 board 状态不变。"""
        return self._search(board, simulations, add_noise)
    def _search(self, board: chess.Board, simulations: int,
                     add_noise: bool) -> Node:
        root = Node()
        work = board.copy()

        for _ in range(simulations):
            path: list[Node] = [root]
            node = root
            local = work.copy()
            value = None

            while True:
                if local.is_game_over(claim_draw=True):
                    value = _terminal_value(local)
                    break

                if not node.expanded:
                    priors, v = self._evaluate(local)
                    if not priors:
                        value = _terminal_value(local)
                        break
                    if add_noise and node is root:
                        priors = self._add_dirichlet(priors)
                    legal = legal_move_indices(local)
                    for i, p in priors.items():
                        node.children[i] = Node(p)
                    node.idx_to_move = legal
                    node.expanded = True
                    value = v
                    break

                idx = self._select(node)
                local.push(node.idx_to_move[idx])
                node = node.children[idx]
                path.append(node)

            # 沿路径回传（子方视角交替取负）
            for depth, nd in enumerate(reversed(path)):
                sign = 1.0 if depth % 2 == 0 else -1.0
                nd.n += 1
                nd.w += sign * value

        return root


def _terminal_value(board: chess.Board) -> float:
    """终局估值，从「当前走子方」视角。"""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.0
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


# ── 选着法 ──────────────────────────────────────────────────


def pick_move(root: Node, temperature: float = 1.0,
              rng: random.Random | None = None) -> tuple[int, chess.Move]:
    """从 MCTS 根节点按 visit count 采样。temperature=0 时取最大值。"""
    items = [(i, ch.n) for i, ch in root.children.items() if ch.n > 0]
    if not items:
        items = [(i, 1) for i in root.children]
    if not items:
        raise RuntimeError("根节点没有任何可选走法")

    if temperature <= 1e-6:
        best = max(items, key=lambda x: x[1])
        return best[0], root.idx_to_move[best[0]]

    counts = np.array([c for _, c in items], dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    probs = counts / counts.sum()
    k = int(np.random.choice(len(items), p=probs))
    idx = items[k][0]
    return idx, root.idx_to_move[idx]


def visit_distribution(root: Node) -> dict[int, float]:
    """根节点的 visit 分布，作为训练用的 policy 标签。"""
    total = sum(ch.n for ch in root.children.values())
    if total == 0:
        n = max(len(root.children), 1)
        return {i: 1.0 / n for i in root.children}
    return {i: ch.n / total for i, ch in root.children.items()}


if __name__ == "__main__":
    import time

    print("=" * 64)
    print("engine.py 自检")
    print("=" * 64)

    b = chess.Board()
    planes = board_to_planes(b)
    assert planes.shape == (19, 8, 8), planes.shape
    print("✓ 棋盘张量形状正确")

    # 走法索引
    legal = legal_move_indices(b)
    print(f"✓ 初始局面合法走法: {len(legal)} 个（应为 20）")
    assert len(legal) == 20, len(legal)

    # 每个走法索引唯一
    assert len(set(legal.keys())) == len(legal)
    print("✓ 走法索引无冲突")

    # 网络 + MCTS
    from net import ChessNet
    net = ChessNet()
    print(f"  网络参数 {net.count_params()/1e6:.2f} M")

    mcts = MCTS(net)
    t = time.time()
    root = mcts.run(b, simulations=20)
    dt = time.time() - t
    print(f"✓ MCTS 20 次模拟耗时 {dt:.2f}s（{dt/20*1000:.0f} ms/次）")

    idx, mv = pick_move(root, temperature=0.0)
    print(f"✓ 选中走法: {mv.uci()}  (index {idx})")

    dist = visit_distribution(root)
    top = sorted(dist.items(), key=lambda x: -x[1])[:3]
    print("  前 3 个走法的访问占比:", [(root.idx_to_move[i].uci(), round(p, 3)) for i, p in top])

    print("=" * 64)
    print("通过")
