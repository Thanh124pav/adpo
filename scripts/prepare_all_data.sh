#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
MODE="${1:---all}"

echo "============================================="
echo " ADPO Data Preparation"
echo " Output: $OUTPUT_DIR"
echo "============================================="

if [[ "$MODE" == "--all" || "$MODE" == "--train-only" ]]; then
    echo ">>> Preparing TRAINING datasets..."
    for ds in gsm8k math numina_math open_math_reasoning aops_instruct big_math_rl fine_math; do
        echo "--- $ds ---"
        python data/prepare_datasets.py --dataset "$ds" --split train --output_dir "$OUTPUT_DIR" \
            || echo "[WARN] Failed: $ds"
    done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--eval-only" ]]; then
    echo ">>> Preparing EVALUATION datasets..."
    for ds in math500 gsm8k_test amc_2023 aime_2024 aime_2025 olympiad_bench minerva_math omni_math hmmt brumo cmimc; do
        echo "--- $ds ---"
        python data/prepare_datasets.py --dataset "$ds" --output_dir "$OUTPUT_DIR" \
            || echo "[WARN] Failed: $ds"
    done
fi

echo "Data preparation complete!"
