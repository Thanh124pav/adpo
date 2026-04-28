#!/usr/bin/env bash
# run_server.sh — run mc_analysis on a SERVER with large GPU (≥16 GB VRAM)
#
# Uses vLLM as backend (auto-starts/stops the server).
# Runs all three tree configs (4-4-4, 6-6-6, 8-8-8) by default.
#
# Requirements:
#   pip install vllm requests numpy pandas pyarrow
#
# Usage:
#   bash run_server.sh [OPTIONS]
#
# Options:
#   --model        deepseek | rho | <full HF model id>  (default: deepseek)
#   --tree         4-4-4 | 6-6-6 | 8-8-8               (default: all three)
#   --question     0 | 1 | 2                             (default: all built-in)
#   --parquet      PATH        load questions from parquet file
#   --num-examples N           random examples from parquet (default: 5)
#   --seed         INT         sampling seed (default: 42)
#   --port         PORT        vLLM server port (default: 8000)
#   --max-len      INT         max_model_len for vLLM (default: 8192)
#   --gpu-mem      FLOAT       gpu_memory_utilization (default: 0.90)
#   --no-server                assume vLLM already running at --port
#   --score-concurrent INT     concurrency for echo/scoring requests (default: 4)
#   --no-compute-p             skip P scoring (recommended for 6-6-6 and larger)
#   --save-dir     DIR         (default: ./results/server/<model>)
#
# Examples:
#   # All trees, built-in questions
#   bash run_server.sh
#
#   # Load 20 random examples from GSM8K parquet, only 6-6-6 tree
#   bash run_server.sh --parquet ./data/processed/train/gsm8k.parquet \
#       --num-examples 20 --tree 6-6-6
#
#   # Rho-Math, MATH dataset, seed 7
#   bash run_server.sh --model rho \
#       --parquet ./data/processed/train/math.parquet \
#       --num-examples 10 --seed 7
#
#   # Server already running on port 8000
#   bash run_server.sh --no-server \
#       --parquet ./data/processed/eval/math500.parquet --num-examples 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
MODEL_KEY="deepseek"
TREE=""
QUESTION=""
PARQUET=""
NUM_EXAMPLES=5
SEED=42
PORT=8000
MAX_LEN=8192
GPU_MEM=0.90
NO_SERVER=0
SCORE_CONCURRENT=4
NO_COMPUTE_P=""
SAVE_DIR=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL_KEY="$2";    shift 2 ;;
    --tree)         TREE="$2";         shift 2 ;;
    --question)     QUESTION="$2";     shift 2 ;;
    --parquet)      PARQUET="$2";      shift 2 ;;
    --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
    --seed)         SEED="$2";         shift 2 ;;
    --port)         PORT="$2";         shift 2 ;;
    --max-len)      MAX_LEN="$2";      shift 2 ;;
    --gpu-mem)      GPU_MEM="$2";      shift 2 ;;
    --no-server)         NO_SERVER=1;            shift   ;;
    --score-concurrent)  SCORE_CONCURRENT="$2";  shift 2 ;;
    --no-compute-p)      NO_COMPUTE_P="--no-compute-p"; shift ;;
    --save-dir)          SAVE_DIR="$2";           shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── model map ─────────────────────────────────────────────────────────────────
case "$MODEL_KEY" in
  deepseek) MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ;;
  rho)      MODEL="microsoft/rho-math-1.1b-v0.1" ;;
  *)        MODEL="$MODEL_KEY" ;;   # accept full HF model id directly
esac

[[ -z "$SAVE_DIR" ]] && SAVE_DIR="./results/server/${MODEL_KEY}"
SERVER_URL="http://localhost:${PORT}/v1"

# ── check vllm ────────────────────────────────────────────────────────────────
command -v vllm &>/dev/null || {
  echo "[error] vllm not found. Install: pip install vllm"
  exit 1
}

# ── build args ────────────────────────────────────────────────────────────────
PY_ARGS=(
  "$SCRIPT_DIR/run_tree_analysis.py"
  --model        "$MODEL"
  --port         "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --save-dir     "$SAVE_DIR"
  --max-tokens        512
  --temperature       0.8
  --max-concurrent    8
  --score-concurrent  "$SCORE_CONCURRENT"
  --num-examples      "$NUM_EXAMPLES"
  --seed         "$SEED"
  # M-token mode by default (no --stop = SPO default)
)

[[ -n "$TREE" ]]         && PY_ARGS+=(--tree "$TREE")
[[ -n "$QUESTION" ]]    && PY_ARGS+=(--question-idx "$QUESTION")
[[ -n "$PARQUET" ]]     && PY_ARGS+=(--parquet "$PARQUET")
[[ -n "$NO_COMPUTE_P" ]] && PY_ARGS+=($NO_COMPUTE_P)
[[ "$NO_SERVER" -eq 1 ]] && PY_ARGS+=(--no-auto-server --server "$SERVER_URL")

echo "============================================================"
echo "  Backend      : vLLM  (port $PORT)"
echo "  Model        : $MODEL"
echo "  Tree         : ${TREE:-all (4-4-4, 6-6-6, 8-8-8)}"
if [[ -n "$PARQUET" ]]; then
  echo "  Questions    : $NUM_EXAMPLES random from $PARQUET (seed=$SEED)"
elif [[ -n "$QUESTION" ]]; then
  echo "  Question     : #$QUESTION"
else
  echo "  Questions    : built-in (3)"
fi
echo "  GPU mem      : $GPU_MEM"
echo "  Max len      : $MAX_LEN"
echo "  Save dir     : $SAVE_DIR"
echo "============================================================"

python3 "${PY_ARGS[@]}"
