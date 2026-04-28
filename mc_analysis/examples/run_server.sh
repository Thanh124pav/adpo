#!/usr/bin/env bash
# run_server.sh — run mc_analysis on a SERVER with large GPU (≥16 GB VRAM)
#
# Uses vLLM as backend (auto-starts/stops the server).
# Runs all three tree configs (4-4-4, 6-6-6, 8-8-8) by default.
#
# Requirements:
#   pip install vllm requests numpy
#
# Usage:
#   bash run_server.sh [deepseek|rho] [OPTIONS]
#
# Options (all optional):
#   --tree       4-4-4 | 6-6-6 | 8-8-8   (default: all three)
#   --question   0 | 1 | 2                (default: all three)
#   --port       PORT                     (default: 8000)
#   --max-len    MAX_MODEL_LEN            (default: 8192)
#   --gpu-mem    GPU_MEMORY_UTILIZATION   (default: 0.90)
#   --no-server                           assume vLLM already running at --port
#
# Examples:
#   bash run_server.sh                          # deepseek, all trees, all questions
#   bash run_server.sh rho --tree 6-6-6         # rho-math, 6-6-6 only
#   bash run_server.sh deepseek --no-server     # server already up on port 8000
#   bash run_server.sh deepseek --gpu-mem 0.95 --max-len 16384
#
# Results saved as JSON + HTML under ./results/server/<model>/

set -euo pipefail

# ── parse args ────────────────────────────────────────────────────────────────
MODEL_KEY="${1:-deepseek}"
shift || true

TREE=""
QUESTION=""
PORT=8000
MAX_LEN=8192
GPU_MEM=0.90
NO_SERVER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree)       TREE="$2";     shift 2 ;;
    --question)   QUESTION="$2"; shift 2 ;;
    --port)       PORT="$2";     shift 2 ;;
    --max-len)    MAX_LEN="$2";  shift 2 ;;
    --gpu-mem)    GPU_MEM="$2";  shift 2 ;;
    --no-server)  NO_SERVER=1;   shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── model map ─────────────────────────────────────────────────────────────────
case "$MODEL_KEY" in
  deepseek) MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ;;
  rho)      MODEL="microsoft/rho-math-1.1b-v0.1" ;;
  *)        echo "Unknown model key '$MODEL_KEY'. Use: deepseek | rho"; exit 1 ;;
esac

SERVER_URL="http://localhost:${PORT}/v1"
SAVE_DIR="./results/server/${MODEL_KEY}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Backend  : vLLM  (server on port $PORT)"
echo "  Model    : $MODEL"
echo "  Tree     : ${TREE:-all (4-4-4, 6-6-6, 8-8-8)}"
echo "  Question : ${QUESTION:-all}"
echo "  GPU mem  : $GPU_MEM"
echo "  Max len  : $MAX_LEN"
echo "  Save dir : $SAVE_DIR"
echo "============================================================"

# ── check vllm ───────────────────────────────────────────────────────────────
command -v vllm &>/dev/null || {
  echo "[error] vllm not found. Install: pip install vllm"
  exit 1
}

# ── build python args ─────────────────────────────────────────────────────────
PY_ARGS=(
  "$SCRIPT_DIR/run_tree_analysis.py"
  --model   "$MODEL"
  --port    "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --save-dir "$SAVE_DIR"
  --max-tokens 512
  --temperature 0.8
  --max-concurrent 16
  # M-token mode by default (no --stop = SPO default)
)

[[ -n "$TREE" ]]     && PY_ARGS+=(--tree "$TREE")
[[ -n "$QUESTION" ]] && PY_ARGS+=(--question-idx "$QUESTION")
[[ "$NO_SERVER" -eq 1 ]] && PY_ARGS+=(--no-auto-server --server "$SERVER_URL")

python3 "${PY_ARGS[@]}"
