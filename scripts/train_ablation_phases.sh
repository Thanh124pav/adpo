#!/bin/bash
# Ablation study: sweep ADPO phase parameters
# 1. Phase detection methods (threshold vs adaptive)
# 2. Percentile values (75, 80, 85, 90)
# 3. Judge types (rule vs vllm)
# 4. Soft assignment sigma (0, 5, 10)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EPOCHS=${EPOCHS:-2}

echo "============================================="
echo " ADPO Ablation: Phase Parameters"
echo "============================================="

# --- Ablation 1: Percentile threshold ---
echo ">>> Ablation 1: Percentile sweep"
for pct in 75.0 80.0 85.0 90.0; do
    echo "--- percentile=$pct ---"
    EXPERIMENT="adpo-ablation-pct${pct}" EPOCHS="$EPOCHS" \
        bash scripts/train_math.sh \
        algorithm.phase_method=adaptive \
        algorithm.phase_percentile="$pct" \
        "$@" || echo "[WARN] pct=$pct failed"
done

# --- Ablation 2: Fixed threshold ---
echo ">>> Ablation 2: Fixed delta sweep"
for delta in 1.0 1.5 2.0 3.0; do
    echo "--- delta=$delta ---"
    EXPERIMENT="adpo-ablation-delta${delta}" EPOCHS="$EPOCHS" \
        bash scripts/train_math.sh \
        algorithm.phase_method=threshold \
        algorithm.phase_delta="$delta" \
        "$@" || echo "[WARN] delta=$delta failed"
done

# --- Ablation 3: Soft assignment sigma ---
echo ">>> Ablation 3: Sigma sweep"
for sigma in 0.0 5.0 10.0 20.0; do
    echo "--- sigma=$sigma ---"
    EXPERIMENT="adpo-ablation-sigma${sigma}" EPOCHS="$EPOCHS" \
        bash scripts/train_math.sh \
        algorithm.phase_sigma="$sigma" \
        "$@" || echo "[WARN] sigma=$sigma failed"
done

# --- Ablation 4: Judge type (if GPU available) ---
echo ">>> Ablation 4: Judge type"
for judge in rule vllm; do
    echo "--- judge=$judge ---"
    EXPERIMENT="adpo-ablation-judge-${judge}" EPOCHS="$EPOCHS" \
        bash scripts/train_math.sh \
        algorithm.judge_type="$judge" \
        "$@" || echo "[WARN] judge=$judge failed"
done

echo "============================================="
echo " Ablation sweep complete!"
echo "============================================="
