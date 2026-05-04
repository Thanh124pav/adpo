#!/usr/bin/env bash
# run_local.sh — run mc_analysis on a LOCAL machine with ~4 GB VRAM
#
# Uses HuggingFace backend (no vLLM server needed) with 4-bit quantization
# so the model fits in 4 GB VRAM.
#
# Requirements:
#   pip install torch transformers bitsandbytes accelerate numpy pandas pyarrow
#
# Usage:
#   bash run_local.sh [OPTIONS]
#
# Options:
#   --model    deepseek | rho | <full HF model id>   (default: deepseek)
#   --tree     4-4-4 | 6-6-6 | 8-8-8                (default: 4-4-4)
#   --question 0 | 1 | 2                             (default: all built-in)
#   --parquet  PATH                                   load questions from file
#   --num-examples N                                  random examples from parquet (default: 5)
#   --seed     INT                                    sampling seed (default: 42)
#   --int8                                            use 8-bit instead of 4-bit
#   --fp16                                            no quantization (needs ≥4 GB free)
#   --save-dir DIR                                    (default: ./results/local/<model>)
#
# Examples:
#   # Built-in questions, 4-4-4 tree
#   bash run_local.sh
#
#   # Load 10 random examples from a parquet dataset
#   bash run_local.sh --parquet ./data/processed/train/gsm8k.parquet --num-examples 10
#
#   # Rho-Math, 6-6-6 tree, parquet, seed 123
#   bash run_local.sh --model rho --tree 6-6-6 \
#       --parquet ./data/processed/train/math.parquet --num-examples 5 --seed 123
#
# Notes:
#   - 4-4-4 tree (85 nodes)  ≈ 5–15 min on a single consumer GPU
#   - 6-6-6 tree (259 nodes) ≈ 20–60 min
#   - 8-8-8 tree (585 nodes) ≈ 1–3 hours

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ─────────────────────────────────────────────────────────────────
MODEL_KEY="deepseek"
TREE="4-4-4"
QUESTION=""
PARQUET=""
NUM_EXAMPLES=5
SEED=42
QUANT="--load-in-4bit"
NO_COMPUTE_P=""
TOP_K=20
SAVE_DIR=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL_KEY="$2";    shift 2 ;;
    --tree)          TREE="$2";         shift 2 ;;
    --question)      QUESTION="$2";     shift 2 ;;
    --parquet)       PARQUET="$2";      shift 2 ;;
    --num-examples)  NUM_EXAMPLES="$2"; shift 2 ;;
    --seed)          SEED="$2";         shift 2 ;;
    --int8)          QUANT="--load-in-8bit"; shift ;;
    --fp16)          QUANT="";          shift ;;
    --no-compute-p)  NO_COMPUTE_P="--no-compute-p"; shift ;;
    --top-k)         TOP_K="$2";        shift 2 ;;
    --save-dir)      SAVE_DIR="$2";     shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── model map ─────────────────────────────────────────────────────────────────
case "$MODEL_KEY" in
  deepseek) MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ;;
  rho)      MODEL="microsoft/rho-math-1.1b-v0.1" ;;
  *)        MODEL="$MODEL_KEY" ;;   # accept full HF model id directly
esac

[[ -z "$SAVE_DIR" ]] && SAVE_DIR="./results/local/${MODEL_KEY}"

# ── check deps ────────────────────────────────────────────────────────────────
python3 -c "import bitsandbytes" 2>/dev/null || {
  if [[ "$QUANT" == "--load-in-4bit" || "$QUANT" == "--load-in-8bit" ]]; then
    echo "[warn] bitsandbytes not found — installing..."
    pip install bitsandbytes accelerate --quiet
  fi
}

# ── build args ────────────────────────────────────────────────────────────────
PY_ARGS=(
  "$SCRIPT_DIR/run_local_hf.py"
  --model          "$MODEL"
  --tree           "$TREE"
  --save-dir       "$SAVE_DIR"
  --num-examples   "$NUM_EXAMPLES"
  --seed           "$SEED"
  --top-k-logprobs "$TOP_K"
)

[[ -n "$QUANT" ]]        && PY_ARGS+=($QUANT)
[[ -n "$QUESTION" ]]     && PY_ARGS+=(--question-idx "$QUESTION")
[[ -n "$PARQUET" ]]      && PY_ARGS+=(--parquet "$PARQUET")
[[ -n "$NO_COMPUTE_P" ]] && PY_ARGS+=($NO_COMPUTE_P)

echo "============================================================"
echo "  Backend      : HuggingFace (${QUANT:-fp16})"
echo "  Model        : $MODEL"
echo "  Tree         : $TREE"
if [[ -n "$PARQUET" ]]; then
  echo "  Questions    : $NUM_EXAMPLES random from $PARQUET (seed=$SEED)"
elif [[ -n "$QUESTION" ]]; then
  echo "  Question     : #$QUESTION"
else
  echo "  Questions    : built-in (3)"
fi
echo "  Save dir     : $SAVE_DIR"
echo "============================================================"

python3 "${PY_ARGS[@]}"
