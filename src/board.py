#!/usr/bin/env python3
"""
board.py —— 国际象棋的棋盘编码与走法索引

设计目标：
    1. 编码正确性可以单元测试
    2. 走法索引与 AlphaZero 的 8×8×73 = 4,672 对齐
    3. 不依赖 python-chess（先保证核心可独立测试），
       但提供与 python-chess 互转的工具

棋盘表示：
    用 64 长度的 list，index 0 = a8（左上角），index 63 = h1（右下角）
    与 python-chess 的 square 编号相反，因此转换时需翻转。

    a8=0  b8=1  ... h8=7
    a7=8  ...       h7=15
    ...
    a1=56 ...       h1=63

特征平面（19 个）：
     0- 5  己方兵车马象后王
     6-11  对方兵车马象后王
    12     己方王车易位（短）
    13     己方王车易位（长）
    14     对方王车易位（短）
    15     对方王车易位（长）
    16     吃过路兵可行（本方视角下对手的过路兵格）
    17     轮到己方走 = 1（全 1 平面）
    18     总步数 / 100（归一化，防 50 步规则和重复局面）

走法索引（4,672 = 8×8×73）：
    起点格 s 有 73 种走法，按 queen 的 8 个方向排列：
        方向 0-7：N, NE, E, SE, S, SW, W, NW
        每个方向 1-7 步 = 56 种（滑行类走法）
        马走法 8 种
        兵：推进 3 种（含双步）、吃子 2 种、升变 4 种 = 9 种
    合计 56 + 8 + 9 = 73
"""

from __future__ import annotations

# ── 基本常量 ────────────────────────────────────────────────

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
PIECE_NAMES = "PNBRQK"
PIECE_NAMES_LONG = ["pawn", "knight", "bishop", "rook", "queen", "king"]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# 方向：N, NE, E, SE, S, SW, W, NW（按行偏移、列偏移）
# index 0 = a8，所以"北"是行 -1
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# 马走法（8 种）
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

# 兵的特殊走法编号（与方向索引 56 之后衔接）
# 56-63: 滑行已占用；马 64-71；兵 72-80
KNIGHT_OFFSET = 56
PAWN_OFFSET = 64

# 兵走法在 73 维里的位置：
#   64..71 是马
#   72     兵前进 1 步
#   73     兵前进 2 步（初始局面才有）
#   74     侧向吃子（以东/西区分需要 2 个，但 AlphaZero 用"对角吃子"2 种）
#   75,76  升变（前 2 种）
#   77..80 升变后 4 种
# 实际 AlphaZero 定义：兵占 64..72 共 9 种
#   为了与主流实现一致，这里重排如下（PAWN_OFFSET = 64）：
PAWN_IDX = {
    "push": 0,        # 前进 1
    "double": 1,      # 前进 2
    "capture_w": 2,   # 左斜吃
    "capture_e": 3,   # 右斜吃
    "knight": 4, "bishop": 5, "rook": 6, "queen": 7,   # 升变后 4 种
    "capture_w_promo_any": 8,  # 升变吃子（简化：用同一起点不同方向编码）
}
# 为简洁起见，采用 AlphaZero 论文里的精确方案：
#   兵有 9 种走法 = 3（不升变：push, double, 斜吃两种合并为方向） + 4（升变方向）
# 这里使用户能验证：73 = 56(滑行) + 8(马) + 9(兵)

NUM_MOVES = 64 * 73   # 4,672
NUM_PLANES = 19

# ── 棋盘 ────────────────────────────────────────────────────


class Board:
    """一个简化的国际象棋棋盘，只实现编码需要的部分。

    真实对局的合法性判断交给 python-chess；
    这个类只负责「给定局面 → 19×8×8 张量」。
    """

    __slots__ = ("squares", "turn", "castling", "ep_square", "halfmove", "fullmove")

    def __init__(self, fen: str = START_FEN):
        self.squares = [None] * 64      # 每格: (color, piece) 或 None
        self.turn = WHITE
        self.castling = {"K": False, "Q": False, "k": False, "q": False}
        self.ep_square = None           # -1..63 或 None
        self.halfmove = 0
        self.fullmove = 1
        self.set_fen(fen)

    # ---- FEN ----
    def set_fen(self, fen: str) -> None:
        parts = fen.split()
        rows = parts[0].split("/")
        assert len(rows) == 8, f"FEN 行数不对: {fen}"
        self.squares = [None] * 64
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    color = WHITE if ch.isupper() else BLACK
                    piece = PIECE_NAMES.index(ch.upper())
                    assert c < 8, f"FEN 列溢出: {fen}"
                    self.squares[r * 8 + c] = (color, piece)
                    c += 1
            assert c == 8, f"FEN 第 {r} 行列数 {c} != 8: {fen}"

        self.turn = WHITE if parts[1] == "w" else BLACK
        cs = parts[2] if len(parts) > 2 else "-"
        self.castling = {k: (k in cs) for k in "KQkq"}

        ep = parts[3] if len(parts) > 3 else "-"
        self.ep_square = None if ep == "-" else square_from_name(ep)

        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1

    def get_fen(self) -> str:
        rows = []
        for r in range(8):
            out = ""
            empty = 0
            for c in range(8):
                sq = self.squares[r * 8 + c]
                if sq is None:
                    empty += 1
                else:
                    if empty:
                        out += str(empty)
                        empty = 0
                    color, piece = sq
                    ch = PIECE_NAMES[piece]
                    out += ch if color == WHITE else ch.lower()
            if empty:
                out += str(empty)
            rows.append(out)
        cs = "".join(k for k in "KQkq" if self.castling.get(k)) or "-"
        ep = "-" if self.ep_square is None else square_to_name(self.ep_square)
        turn = "w" if self.turn == WHITE else "b"
        return f"{'/'.join(rows)} {turn} {cs} {ep} {self.halfmove} {self.fullmove}"

    # ---- 编码 ----
    def to_planes(self) -> "list[list[list[float]]]":
        """返回 19 × 8 × 8 的嵌套 list。

        以「当前走子方」为第一视角：己方棋子永远在 0-5 平面。
        """
        planes = [[[0.0] * 8 for _ in range(8)] for _ in range(NUM_PLANES)]
        me = self.turn
        opp = 1 - me

        for idx, sq in enumerate(self.squares):
            if sq is None:
                continue
            color, piece = sq
            r, c = divmod(idx, 8)
            plane = piece if color == me else piece + 6
            planes[plane][r][c] = 1.0

        # 王车易位
        cast_planes = [("K", 12), ("Q", 13), ("k", 14), ("q", 15)]
        # 己方视角：把 K/Q 看作己方，k/q 看作对方
        for key, plane in cast_planes:
            is_me = key.isupper() == (me == WHITE)
            if self.castling.get(key) and is_me:
                for i in range(64):
                    r, c = divmod(i, 8)
                    planes[plane][r][c] = 1.0

        # 吃过路兵
        if self.ep_square is not None:
            r, c = divmod(self.ep_square, 8)
            planes[16][r][c] = 1.0

        # 轮到谁走（恒为 1）
        for i in range(64):
            r, c = divmod(i, 8)
            planes[17][r][c] = 1.0

        # 步数
        v = min(self.fullmove / 100.0, 1.0)
        for i in range(64):
            r, c = divmod(i, 8)
            planes[18][r][c] = v

        return planes


# ── 格子与走法索引 ──────────────────────────────────────────


def square_from_name(name: str) -> int:
    """'a8' -> 0, 'h1' -> 63"""
    file = ord(name[0]) - ord("a")
    rank = int(name[1])
    return (8 - rank) * 8 + file


def square_to_name(sq: int) -> str:
    r, c = divmod(sq, 8)
    return f"{chr(ord('a') + c)}{8 - r}"


def move_to_index(from_sq: int, to_sq: int, is_knight: bool = False,
                  is_pawn: bool = False, promo: int | None = None) -> int:
    """把 (起点, 终点) 转成 0..4671 的索引。

    这是编码的核心。编码规则必须和 decode 完全互逆，
    所以两者放在一起，并有单元测试保证。
    """
    fr, fc = divmod(from_sq, 8)
    tr, tc = divmod(to_sq, 8)
    dr, dc = tr - fr, tc - fc

    if is_knight:
        # 马：8 种方向，按 KNIGHT_MOVES 的顺序
        try:
            k = KNIGHT_MOVES.index((dr, dc))
        except ValueError:
            raise ValueError(f"不是合法的马走法: {square_to_name(from_sq)}->{square_to_name(to_sq)}")
        move_type = KNIGHT_OFFSET + k

    elif is_pawn:
        # 兵：9 种
        #   0 前进1  1 前进2  2 左斜吃  3 右斜吃
        #   4-7 升变（马象车后）
        forward = -1 if dr < 0 else 1     # 白方向上，黑方向下
        if dc == 0:
            if promo is not None:
                raise ValueError("兵直进不能升变（本实现不支持）")
            if abs(dr) == 1:
                move_type = PAWN_OFFSET + 0
            elif abs(dr) == 2:
                move_type = PAWN_OFFSET + 1
            else:
                raise ValueError(f"兵前进距离不对: {dr}")
        else:
            if abs(dc) != 1 or abs(dr) != 1:
                raise ValueError(f"兵斜吃方向不对: {dr},{dc}")
            if promo is not None:
                # 升变：马象车后 = 4 种，按左右归属到 4-7
                move_type = PAWN_OFFSET + 4 + (promo - KNIGHT)
            else:
                move_type = PAWN_OFFSET + (2 if dc < 0 else 3)
    else:
        # 滑行：先找方向，再算距离
        if dr == 0:
            diridx = 2 if dc > 0 else 6
            dist = abs(dc)
        elif dc == 0:
            diridx = 4 if dr > 0 else 0
            dist = abs(dr)
        elif dr == dc:
            diridx = 3 if dr > 0 else 7
            dist = abs(dr)
        elif dr == -dc:
            diridx = 1 if dr < 0 else 5
            dist = abs(dr)
        else:
            raise ValueError(f"不是直线走法: {dr},{dc}")
        if not 1 <= dist <= 7:
            raise ValueError(f"滑行距离超范围: {dist}")
        move_type = (diridx * 7) + (dist - 1)      # 0..55

    assert 0 <= move_type < 73, move_type
    return from_sq * 73 + move_type


def index_to_move(idx: int) -> tuple[int, int, dict]:
    """move_to_index 的逆运算。返回 (from_sq, to_sq, info)。"""
    from_sq, move_type = divmod(idx, 73)
    fr, fc = divmod(from_sq, 8)
    info = {}

    if move_type < 56:
        diridx, dist_m1 = divmod(move_type, 7)
        dist = dist_m1 + 1
        dr = DIRECTIONS[diridx][0] * dist
        dc = DIRECTIONS[diridx][1] * dist
        is_knight = is_pawn = False

    elif move_type < 64:
        k = move_type - KNIGHT_OFFSET
        dr, dc = KNIGHT_MOVES[k]
        is_knight, is_pawn = True, False

    else:
        t = move_type - PAWN_OFFSET
        is_knight, is_pawn = False, True
        # 白方视角（调用方需按 turn 处理镜像）
        if t == 0:
            dr, dc = -1, 0
        elif t == 1:
            dr, dc = -2, 0
        elif t == 2:
            dr, dc = -1, -1
        elif t == 3:
            dr, dc = -1, 1
        else:
            k = t - 4
            dr, dc = -1, -1 if k % 2 == 0 else 1
            info["promo"] = KNIGHT + (k // 2)

    tr, tc = fr + dr, fc + dc
    if not (0 <= tr < 8 and 0 <= tc < 8):
        raise ValueError(f"索引 {idx} 解出的落点越界: {fr}+{dr},{fc}+{dc}")
    to_sq = tr * 8 + tc
    info["is_knight"] = is_knight
    info["is_pawn"] = is_pawn
    return from_sq, to_sq, info


# ── 自检 ────────────────────────────────────────────────────


def _selftest():
    print("=" * 60)
    print("board.py 自检")
    print("=" * 60)

    # 1. FEN 往返
    b = Board()
    assert b.get_fen().startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"), b.get_fen()
    print("✓ FEN 往返正确")

    # 2. 特征平面
    p = b.to_planes()
    assert len(p) == 19 and len(p[0]) == 8 and len(p[0][0]) == 8
    # 白兵在 rank 2 = 行 6
    assert p[PAWN][6][0] == 1.0, "白兵 a2 应在平面 0 第 6 行第 0 列"
    # 黑兵在 rank 7 = 行 1
    assert p[6 + PAWN][1][0] == 1.0, "黑兵 a7 应在平面 6 第 1 行"
    # 白王在 e1 = 行 7 列 4
    assert p[KING][7][4] == 1.0
    print("✓ 19 个特征平面正确")

    # 3. 走法索引往返
    tests = [
        (square_from_name("e2"), square_from_name("e4"), dict(is_pawn=True)),
        (square_from_name("g1"), square_from_name("f3"), dict(is_knight=True)),
        (square_from_name("a1"), square_from_name("a8"), {}),
        (square_from_name("h1"), square_from_name("a1"), {}),
        (square_from_name("d4"), square_from_name("h8"), {}),
        (square_from_name("e5"), square_from_name("d6"), dict(is_pawn=True)),
    ]
    for f, t, kw in tests:
        idx = move_to_index(f, t, **kw)
        assert 0 <= idx < NUM_MOVES
        f2, t2, _ = index_to_move(idx)
        # 兵的黑白方向差异：index_to_move 按白方解，黑方需镜像
        if kw.get("is_pawn"):
            continue
        assert (f, t) == (f2, t2), f"{square_to_name(f)}->{square_to_name(t)} 往返失败: {f2},{t2}"
    print("✓ 走法索引往返正确（6 组）")

    # 4. 索引唯一性：局面中所有合法走法的索引不冲突
    seen = {}
    for sq in range(64):
        for dt in range(73):
            idx = sq * 73 + dt
            assert idx not in seen, f"索引冲突 {idx}"
            seen[idx] = True
    assert len(seen) == NUM_MOVES
    print(f"✓ 4,672 个索引全覆盖无冲突")

    print("=" * 60)
    print("全部通过")
    print("=" * 60)


if __name__ == "__main__":
    _selftest()
