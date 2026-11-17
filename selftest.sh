#!/bin/bash
# 自测脚本：让模型看几个典型局面，检查走法是否合理
E="./benzene"
run() {
  local name="$1" pos="$2" ms="${3:-6000}"
  printf 'position %s\ngo movetime %s\nquit\n' "$pos" "$ms" | $E 2>/dev/null \
    | grep -E "^info|^bestmove" | tail -2 \
    | sed "s/^/[$name] /"
}

echo "=== Benzene-0.2-Phenol 局面测试 ==="
run "开局      " "startpos"
run "意大利    " "startpos moves e2e4 e7e5 g1f3 b8c6 f1c4" 8000
run "西西里    " "startpos moves e2e4 c7c5 g1f3 d7d6 d2d4 c5d4" 8000
run "后翼弃兵  " "startpos moves d2d4 d7d5 c2c4" 8000
run "残局      " "fen 8/8/8/4k3/8/8/4P3/4K3 w - - 0 1" 5000
run "战术-捉双 " "fen r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1" 8000
