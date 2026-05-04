#!/usr/bin/env bash
# run_api.sh — run mc_analysis against an external OpenAI-compatible API endpoint
#
# No local GPU or vLLM server needed.
# P is computed inline via answer branch (echo mode unsupported by most APIs).
#
# Requirements:
#   pip install requests numpy pandas pyarrow
#
# Usage:
#   bash run_api.sh [OPTIONS]
#
# Options:
#   --api-url      URL          API base URL (required)
#   --api-key      KEY          API key (or set API_KEY env var)
#   --model        ID           Full model ID on the API (required)
#   --tree         4-4-4 | 6-6-6 | 8-8-8   (default: all three)
#   --question     0 | 1 | 2                (default: all built-in)
#   --parquet      PATH         load questions from parquet file
#   --num-examples N            random examples from parquet (default: 5)
#   --seed         INT          sampling seed (default: 42)
#   --max-tokens   INT          tokens per step (default: 512)
#   --max-concurrent INT        generation concurrency — keep low for
#                               rate-limited APIs (default: 4)
#   --score-concurrent INT      JSD annotation concurrency (default: 2)
#   --no-compute-p-inline       disable inline P (skip P entirely)
#   --save-dir     DIR          (default: ./results/api/<model>)
#
# Provider quick-start
# --------------------
#   # Together AI
#   export API_KEY=$TOGETHER_API_KEY
#   bash run_api.sh \
#       --api-url https://api.together.xyz/v1 \
#       --model   deepseek-ai/DeepSeek-R1 \
#       --tree    4-4-4
#
#   # OpenAI
#   export API_KEY=$OPENAI_API_KEY
#   bash run_api.sh \
#       --api-url https://api.openai.com/v1 \
#       --model   gpt-4o-mini \
#       --parquet ./data/processed/train/gsm8k.parquet \
#       --num-examples 10
#
#   # Local vLLM (no key needed)
#   bash run_api.sh \
#       --api-url http://localhost:8000/v1 \
#       --model   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
API_URL=""
API_KEY="${API_KEY:-}"
MODEL=""
TREE=""
QUESTION=""
PARQUET=""
NUM_EXAMPLES=5
SEED=42
MAX_TOKENS=512
MAX_CONCURRENT=4
SCORE_CONCURRENT=2
NO_COMPUTE_P_INLINE=""
SAVE_DIR=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-url)            API_URL="$2";          shift 2 ;;
    --api-key)            API_KEY="$2";          shift 2 ;;
    --model)              MODEL="$2";            shift 2 ;;
    --tree)               TREE="$2";             shift 2 ;;
    --question)           QUESTION="$2";         shift 2 ;;
    --parquet)            PARQUET="$2";          shift 2 ;;
    --num-examples)       NUM_EXAMPLES="$2";     shift 2 ;;
    --seed)               SEED="$2";             shift 2 ;;
    --max-tokens)         MAX_TOKENS="$2";       shift 2 ;;
    --max-concurrent)     MAX_CONCURRENT="$2";   shift 2 ;;
    --score-concurrent)   SCORE_CONCURRENT="$2"; shift 2 ;;
    --no-compute-p-inline) NO_COMPUTE_P_INLINE="--no-compute-p-inline"; shift ;;
    --save-dir)           SAVE_DIR="$2";         shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── validate ──────────────────────────────────────────────────────────────────
[[ -z "$API_URL" ]] && { echo "[error] --api-url is required"; exit 1; }
[[ -z "$MODEL"   ]] && { echo "[error] --model is required";   exit 1; }

# Derive a short name for the save directory
MODEL_SLUG="${MODEL##*/}"   # last component after /
[[ -z "$SAVE_DIR" ]] && SAVE_DIR="./results/api/${MODEL_SLUG}"

# ── build args ────────────────────────────────────────────────────────────────
PY_ARGS=(
  "$SCRIPT_DIR/run_api.py"
  --api-url          "$API_URL"
  --model            "$MODEL"
  --save-dir         "$SAVE_DIR"
  --max-tokens       "$MAX_TOKENS"
  --temperature      0.8
  --max-concurrent   "$MAX_CONCURRENT"
  --score-concurrent "$SCORE_CONCURRENT"
  --num-examples     "$NUM_EXAMPLES"
  --seed             "$SEED"
)

[[ -n "$API_KEY" ]]             && PY_ARGS+=(--api-key "$API_KEY")
[[ -n "$TREE" ]]                && PY_ARGS+=(--tree "$TREE")
[[ -n "$QUESTION" ]]            && PY_ARGS+=(--question-idx "$QUESTION")
[[ -n "$PARQUET" ]]             && PY_ARGS+=(--parquet "$PARQUET")
[[ -n "$NO_COMPUTE_P_INLINE" ]] && PY_ARGS+=($NO_COMPUTE_P_INLINE)

echo "============================================================"
echo "  API URL      : $API_URL"
echo "  Model        : $MODEL"
echo "  Tree         : ${TREE:-all (4-4-4, 6-6-6, 8-8-8)}"
if [[ -n "$PARQUET" ]]; then
  echo "  Questions    : $NUM_EXAMPLES random from $PARQUET (seed=$SEED)"
elif [[ -n "$QUESTION" ]]; then
  echo "  Question     : #$QUESTION"
else
  echo "  Questions    : built-in (3)"
fi
echo "  Max concurr  : $MAX_CONCURRENT"
echo "  Save dir     : $SAVE_DIR"
echo "============================================================"

python3 "${PY_ARGS[@]}"
