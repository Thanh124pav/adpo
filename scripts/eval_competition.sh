#!/bin/bash
# Evaluate on competition math benchmarks
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_PATH=""
N_SAMPLES=${N_SAMPLES:-1}
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_TOKENS=${MAX_TOKENS:-2048}
TP_SIZE=${TP_SIZE:-1}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --n_samples) N_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --tp_size) TP_SIZE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash scripts/eval_competition.sh --model_path <path>"
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="results/${MODEL_NAME}/competition"

echo "============================================="
echo " ADPO Competition Math Evaluation"
echo " Model: $MODEL_PATH"
echo "============================================="

python evaluation/evaluate.py \
    --model_path "$MODEL_PATH" \
    --datasets amc_2023 aime_2024 aime_2025 hmmt brumo cmimc \
    --data_dir data/processed/eval \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --tensor_parallel_size "$TP_SIZE"
