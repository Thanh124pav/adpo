#!/bin/bash
# Evaluate with pass@N and maj@N metrics
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_PATH=""
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-0.7}
MAX_TOKENS=${MAX_TOKENS:-2048}
TP_SIZE=${TP_SIZE:-1}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --n) N_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --tp_size) TP_SIZE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash scripts/eval_passn.sh --model_path <path> --n 8"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="results/${MODEL_NAME}/pass_at_${N_SAMPLES}"

echo "============================================="
echo " ADPO pass@$N_SAMPLES / maj@$N_SAMPLES"
echo " Model: $MODEL_PATH"
echo "============================================="

python evaluation/evaluate.py \
    --model_path "$MODEL_PATH" \
    --datasets math500 gsm8k_test aime_2024 aime_2025 \
    --data_dir data/processed/eval \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --tensor_parallel_size "$TP_SIZE"
