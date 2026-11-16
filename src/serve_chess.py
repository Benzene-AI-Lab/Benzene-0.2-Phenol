#!/usr/bin/env python3
"""
serve_chess.py —— 对弈界面后端

提供：
    GET  /                → 网页
    POST /move            → 你走一步，模型回一步
    POST /new             → 新开一局
    GET  /state           → 当前局面

模型推理：
    MCTS 次数可调。CPU 上 al kane 网络约 29 ms/次模拟：
        100 次  → 3 秒/步   （约 1000 Elo）
        400 次  → 12 秒/步  （约 1300 Elo）
        1200 次 → 35 秒/步  （约 1500 Elo）

用法：
    python serve_chess.py --ckpt ../weights/0.2-init-alkane-best.pt
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import chess
import chess.pgn
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MCTS, board_to_planes, visit_distribution, pick_move
from net import build as build_net


# ── 全局状态 ────────────────────────────────────────────────

class GameState:
    """单盘对局的状态。

    每开一局生成一个新的 game_id，历史对局保存在 self.history 里，
    导出 PGN 时可以按 game_id 取任意一盘。
    """

    def __init__(self):
        self.board = chess.Board()
        self.lock = threading.Lock()
        self.mcts: MCTS | None = None
        self.human_color = chess.WHITE
        self.last_model_info: dict = {}
        self.game_id = ""
        self.history: dict[str, dict] = {}      # game_id → {pgn, moves, result, ts}
        self._reset_pgn()

    def _reset_pgn(self):
        g = chess.pgn.Game()
        g.headers["Event"] = "Benzene 0.2 Playground"
        g.headers["Site"] = "127.0.0.1"
        g.headers["Date"] = time.strftime("%Y.%m.%d")
        g.headers["Time"] = time.strftime("%H:%M:%S")
        g.headers["White"] = "Human" if self.human_color == chess.WHITE else "Benzene"
        g.headers["Black"] = "Benzene" if self.human_color == chess.WHITE else "Human"
        g.headers["Result"] = "*"
        self.pgn_game = g
        self.game_id = time.strftime("%Y%m%d-%H%M%S") + f"-{len(self.history)+1}"

    def reset(self, human_color: chess.Color = chess.WHITE):
        with self.lock:
            # 把上一局存进历史（如果有走法）
            if self.pgn_game is not None and self.board.move_stack:
                self._archive()
            self.board = chess.Board()
            self.human_color = human_color
            self.last_model_info = {}
            self._reset_pgn()

    def _archive(self):
        self.history[self.game_id] = {
            "pgn": self.to_pgn(self.game_id),
            "moves": self.board.ply(),
            "result": self.board.result(claim_draw=True) if self.board.is_game_over(claim_draw=True) else "*",
            "ts": time.time(),
        }

    def to_pgn(self, game_id: str = "") -> str:
        """导出指定对局的 PGN；game_id 为空则导出当前这盘。"""
        if game_id and game_id in self.history:
            return self.history[game_id]["pgn"]
        if self.pgn_game is None:
            return ""
        import io
        game = self.pgn_game
        node = game
        for mv in self.board.move_stack:
            node = node.add_variation(mv)
        if self.board.is_game_over(claim_draw=True):
            game.headers["Result"] = self.board.result(claim_draw=True)
        buf = io.StringIO()
        print(game, file=buf, end="\n\n")
        return buf.getvalue()

    def list_games(self) -> list[dict]:
        """返回历史 + 当前对局的列表。"""
        out = []
        for gid, d in sorted(self.history.items(), key=lambda x: -x[1]["ts"]):
            out.append({"id": gid, "moves": d["moves"], "result": d["result"],
                        "current": False})
        # 当前局
        cur_moves = self.board.ply()
        if cur_moves or not out:
            out.insert(0, {
                "id": self.game_id,
                "moves": cur_moves,
                "result": self.board.result(claim_draw=True) if self.board.is_game_over(claim_draw=True) else "*",
                "current": True,
            })
        return out


STATE = GameState()
CONFIG = {"sims": 400, "temperature": 0.0, "net": None}


# ── 页面 ────────────────────────────────────────────────────

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Benzene 0.2 · __MODEL_TITLE__</title>
<link rel="stylesheet"
 href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "PingFang SC", sans-serif; background:#1a1a1a;
         color:#e8e8e8; margin:0; padding:20px;
         display:flex; gap:28px; justify-content:center; align-items:flex-start;
         min-height:100vh; }
  #board { width: 560px; }
  .panel { width: 320px; }
  h1 { font-size:16px; font-weight:600; margin:0 0 4px; letter-spacing:.02em; }
  .sub { font-size:11px; color:#888; margin-bottom:20px; }
  .row { margin-bottom:14px; }
  label { display:block; font-size:11px; color:#999; margin-bottom:5px;
          text-transform:uppercase; letter-spacing:.06em; }
  select, button { width:100%; padding:8px 10px; border-radius:6px;
         border:1px solid #333; background:#252525; color:#e8e8e8;
         font-size:13px; font-family:inherit; cursor:pointer; }
  button:hover:not(:disabled) { background:#2f2f2f; border-color:#444; }
  button:disabled { opacity:.4; cursor:default; }
  button.primary { background:#2d6a4f; border-color:#2d6a4f; }
  button.primary:hover:not(:disabled) { background:#357d5c; }
  #status { font-size:12px; padding:10px; border-radius:6px; background:#252525;
            min-height:40px; line-height:1.6; }
  #status.thinking { background:#2a2412; color:#e6c86e; }
  #status.over { background:#2a1a1a; color:#e88; }
  .stat { display:flex; justify-content:space-between; font-size:11px;
          color:#888; padding:4px 0; border-bottom:1px solid #262626; }
  .stat b { color:#ccc; font-weight:500; }
  #pv { font-family:ui-monospace, monospace; font-size:11px; color:#7fa;
        height:32px; overflow:hidden; margin-top:6px; word-break:break-all; }
  .divider { height:1px; background:#262626; margin:18px 0; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  /* 强制棋子图片可见 */
  .piece-417db { background-size: cover !important; }
  .board-b72b1 .piece-417db { width:100%; height:100%; background-repeat:no-repeat; }
  [class*="piece-"] { background-size: 100% 100% !important; }
</style>
</head>
<body>

<div id="board"></div>

<div class="panel">
  <h1>Benzene 0.2 · __MODEL_TITLE__</h1>
  <div class="sub">国际象棋 · 从零训练</div>

  <div class="row">
    <label>难度（MCTS 模拟次数）</label>
    <select id="sims">
      <option value="200">快 · 200 次（约 2 秒/步）</option>
      <option value="800" selected>中 · 800 次（约 7 秒/步）</option>
      <option value="1600">难 · 1600 次（约 13 秒/步）</option>
      <option value="3200">极难 · 3200 次（约 27 秒/步）</option>
    </select>
  </div>

  <div class="row">
    <label>执方</label>
    <select id="color">
      <option value="w" selected>白棋（先手）</option>
      <option value="b">黑棋（后手）</option>
    </select>
  </div>

  <div class="row">
    <button class="primary" id="newGame">新开一局</button>
  </div>

  <div class="row">
    <label>导出哪一局</label>
    <select id="gameSel"></select>
  </div>
  <div class="row grid2">
    <button id="dlPgn">下载 PGN</button>
    <button id="copyPgn">复制 PGN</button>
  </div>

  <div class="row">
    <div id="status">准备就绪。走一步棋开始。</div>
  </div>

  <div class="divider"></div>

  <div class="stat"><span>局面评估</span><b id="s-eval">—</b></div>
  <div class="stat"><span>思考耗时</span><b id="s-time">—</b></div>
  <div class="stat"><span>搜索节点</span><b id="s-nodes">—</b></div>
  <div class="stat"><span>已走步数</span><b id="s-plies">0</b></div>
  <div id="pv"></div>
</div>

<script src="https://unpkg.com/jquery@3.7.1/dist/jquery.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script>
// 用 Unicode 棋子字符生成 SVG data URI —— 不依赖任何外部图片
const GLYPH = {k:'\u265A', q:'\u265B', r:'\u265C', b:'\u265D', n:'\u265E', p:'\u265F'};
const PIECE_IMG = {};
for (const color of ['w','b']) {
  for (const [t, g] of Object.entries(GLYPH)) {
    const fill = color === 'w' ? '#fff' : '#1a1a1a';
    const stroke = color === 'w' ? '#1a1a1a' : '#000';
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45">`
      + `<text x="22.5" y="35" font-size="40" text-anchor="middle" `
      + `fill="${fill}" stroke="${stroke}" stroke-width="1.2" `
      + `font-family="Apple Symbols, Segoe UI Symbol, sans-serif">${g}</text></svg>`;
    PIECE_IMG[color + t.toUpperCase()] = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svg)));
  }
}

const API = location.origin;
let board, gameFen = null, humanColor = 'w', busy = false, myTurn = true;

function setStatus(t, cls) {
  const el = document.getElementById('status');
  el.textContent = t;
  el.className = cls || '';
}

function refreshStats(info) {
  if (info.eval !== undefined) {
    const v = info.eval;
    document.getElementById('s-eval').textContent =
      (v > 0 ? '+' : '') + v.toFixed(2) + '  ' + (v > 0.3 ? '白优' : v < -0.3 ? '黑优' : '均势');
  }
  if (info.ms !== undefined) document.getElementById('s-time').textContent = (info.ms/1000).toFixed(1) + ' 秒';
  if (info.nodes !== undefined) document.getElementById('s-nodes').textContent = info.nodes;
  if (info.pv) document.getElementById('pv').textContent = '主变: ' + info.pv;
}

async function refreshGames() {
  try {
    const d = await (await fetch(API + '/games')).json();
    const sel = document.getElementById('gameSel');
    const prev = sel.value;
    sel.innerHTML = '';
    for (const g of d.games) {
      const o = document.createElement('option');
      o.value = g.id;
      o.textContent = `${g.current ? '● 当前' : g.id} · ${g.moves} 步 · ${g.result}`;
      sel.appendChild(o);
    }
    if (prev) sel.value = prev;
  } catch(e) {}
}

async function newGame() {
  humanColor = document.getElementById('color').value;
  const res = await fetch(API + '/new', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ human: humanColor })
  });
  const d = await res.json();
  gameFen = d.fen;
  board.position(gameFen, false);
  board.orientation(humanColor === 'w' ? 'white' : 'black');
  document.getElementById('s-plies').textContent = '0';
  setStatus(humanColor === 'w' ? '轮到你走。' : '模型先手思考中…');
  busy = false;
  myTurn = (humanColor === 'w');
  await refreshGames();
  if (humanColor === 'b') await modelMove();
}

async function modelMove() {
  if (busy) return;
  busy = true;
  setStatus('模型思考中…（' + document.getElementById('sims').value + ' 次模拟）', 'thinking');
  try {
    const res = await fetch(API + '/move', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ sims: +document.getElementById('sims').value })
    });
    const d = await res.json();
    if (d.error) { setStatus('错误: ' + d.error, 'over'); busy = false; return; }
    if (d.model_move) {
      board.position(d.fen, true);
      refreshStats(d.info);
    }
    gameFen = d.fen;
    document.getElementById('s-plies').textContent = d.plies;
    if (d.gameover) {
      setStatus(d.result_text, 'over');
      myTurn = false;
      refreshGames();
    } else {
      setStatus('轮到你走。');
      myTurn = true;
    }
  } catch(e) {
    setStatus('请求失败: ' + e.message, 'over');
  }
  busy = false;
}

async function onDrop(source, target) {
  if (busy) return 'snapback';
  if (!myTurn) { setStatus('等模型走完再动。'); return 'snapback'; }

  const res = await fetch(API + '/human', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ from: source, to: target, promotion: (target[1]==='1'||target[1]==='8') ? 'q' : '' })
  });
  const d = await res.json();
  if (!d.ok) {
    setStatus(d.error || '这步不合法', '');
    return 'snapback';
  }
  gameFen = d.fen;
  myTurn = false;
  board.position(d.fen, true);
  document.getElementById('s-plies').textContent = d.plies;
  if (d.gameover) { setStatus(d.result_text, 'over'); return; }
  setTimeout(modelMove, 150);
}

$(function(){
  console.log('PIECE_IMG keys:', Object.keys(PIECE_IMG).length, 'wP sample:', PIECE_IMG.wP?.slice(0,60));
  board = Chessboard('board', {
    draggable: true,
    position: 'start',
    pieceTheme: function(piece) { return PIECE_IMG[piece]; },
    onDrop: onDrop
  });
  document.getElementById('newGame').onclick = newGame;
  refreshGames();
  document.getElementById('dlPgn').onclick = () => {
    const id = document.getElementById('gameSel').value;
    location.href = API + '/pgn?id=' + encodeURIComponent(id);
  };
  document.getElementById('copyPgn').onclick = async () => {
    try {
      const id = document.getElementById('gameSel').value;
      const t = await (await fetch(API + '/pgn?id=' + encodeURIComponent(id))).text();
      await navigator.clipboard.writeText(t);
      setStatus('PGN 已复制到剪贴板。');
    } catch(e) { setStatus('复制失败: ' + e.message, 'over'); }
  };
  document.getElementById('color').onchange = newGame;
  $(window).resize(() => board.resize());
  newGame();
});
</script>
</body>
</html>
"""


# ── HTTP 处理 ───────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text: str):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/games":
            return self._json({"games": STATE.list_games(), "current": STATE.game_id})
        if path == "/pgn":
            qs = urlparse(self.path).query
            gid = ""
            for kv in qs.split("&"):
                if kv.startswith("id="):
                    gid = kv[3:]
            pgn = STATE.to_pgn(gid)
            body = pgn.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-chess-pgn; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="benzene_game.pgn"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            title = CONFIG.get("title", "对弈")
            self._html(PAGE.replace("__MODEL_TITLE__", title))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if path == "/new":
            human = payload.get("human", "w")
            STATE.reset(chess.WHITE if human == "w" else chess.BLACK)
            return self._json({"fen": STATE.board.fen()})

        if path == "/human":
            with STATE.lock:
                b = STATE.board
                try:
                    f, t = payload["from"], payload["to"]
                    promo = payload.get("promotion") or ""
                    # 只有走到最后一行（升变）才需要后缀，否则拼上会变成非法 UCI
                    piece = STATE.board.piece_at(chess.parse_square(f))
                    need_promo = (piece is not None
                                  and piece.piece_type == chess.PAWN
                                  and t[1] in ("1", "8"))
                    uci = f + t + ((promo or "q") if need_promo else "")
                    mv = chess.Move.from_uci(uci)
                except (KeyError, ValueError):
                    return self._json({"ok": False, "error": "走法格式错误"})
                if mv not in b.legal_moves:
                    return self._json({"ok": False, "error": "不合法"})
                b.push(mv)
                over = b.is_game_over(claim_draw=True)
                return self._json({
                    "ok": True, "fen": b.fen(), "plies": b.ply(),
                    "gameover": over,
                    "result_text": _result_text(b) if over else "",
                })

        if path == "/move":
            sims = int(payload.get("sims", 400))
            with STATE.lock:
                b = STATE.board
                if b.is_game_over(claim_draw=True):
                    return self._json({"fen": b.fen(), "plies": b.ply(),
                                       "gameover": True, "result_text": _result_text(b)})
                if not CONFIG["net"]:
                    return self._json({"error": "模型未加载"})

                t0 = time.time()
                root = STATE.mcts.run(b, simulations=sims)
                _, mv = pick_move(root, temperature=CONFIG["temperature"])
                dist = visit_distribution(root)
                top = sorted(dist.items(), key=lambda x: -x[1])[:3]
                pv_moves = [root.idx_to_move[i].uci() for i, _ in top]
                b.push(mv)
                ms = (time.time() - t0) * 1000

                total_nodes = sum(ch.n for ch in root.children.values())
                # 根节点 value 是当前走子方视角
                raw_v = root.w / max(root.n, 1)

                over = b.is_game_over(claim_draw=True)
                return self._json({
                    "fen": b.fen(), "plies": b.ply(),
                    "model_move": mv.uci(),
                    "gameover": over,
                    "result_text": _result_text(b) if over else "",
                    "info": {
                        "eval": raw_v, "ms": round(ms), "nodes": total_nodes,
                        "pv": " ".join(pv_moves),
                    },
                })

        self._json({"error": "not found"}, 404)


def _result_text(b: chess.Board) -> str:
    o = b.outcome(claim_draw=True)
    if o is None:
        return "对局结束"
    names = {
        chess.Termination.CHECKMATE: "将杀",
        chess.Termination.STALEMATE: "逼和",
        chess.Termination.INSUFFICIENT_MATERIAL: "子力不足",
        chess.Termination.SEVENTYFIVE_MOVES: "75 步规则",
        chess.Termination.FIVEFOLD_REPETITION: "五次重复",
        chess.Termination.FIFTY_MOVES: "50 步规则",
        chess.Termination.THREEFOLD_REPETITION: "三次重复",
    }
    why = names.get(o.termination, o.termination.name)
    if o.winner is None:
        return f"和棋（{why}）"
    return f"{'白方' if o.winner == chess.WHITE else '黑方'}胜（{why}）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../weights/0.2-init-alkane-best.pt")
    ap.add_argument("--scale", default="alkane")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ck_path = Path(args.ckpt)
    if not ck_path.exists():
        print(f"找不到权重 {ck_path}")
        return

    # 单线程最快：网络小，多线程的调度开销 > 并行收益
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ck = torch.load(str(ck_path), map_location=args.device, weights_only=False)
    scale = ck.get("config", {}).get("scale", args.scale)
    net = build_net(scale).to(args.device)
    net.load_state_dict(ck["model"])
    net.eval()

    print("=" * 56)
    print(f"权重      {ck_path.name}")
    print(f"规模      {scale}  ({net.count_params()/1e6:.2f} M)")
    print(f"训练      epoch {ck.get('epoch','?')} · val {ck.get('val_loss',0):.4f} "
          f"· top1 {ck.get('top1',0)*100:.1f}%")
    print(f"默认难度  {args.sims} 次模拟")
    print(f"地址      http://127.0.0.1:{args.port}")
    print("=" * 56)

    CONFIG["net"] = net
    CONFIG["sims"] = args.sims
    # 标题：从权重文件名推断分子名
    fname = ck_path.name.lower()
    mol = "Alkane" if "alkane" in fname else ("Phenol" if "phenol" in fname else ("Alkene" if "alkene" in fname else "Alkyne" if "alkyne" in fname else scale.capitalize()))
    CONFIG["title"] = f"{mol} · {net.count_params()/1e6:.2f}M"
    STATE.mcts = MCTS(net, device=args.device)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
