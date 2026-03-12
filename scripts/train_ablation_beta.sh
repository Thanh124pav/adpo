#!/bin/bash
# Ablation study: sweep ADPO beta values
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

for beta in 0.0 0.5 1.0 2.0; do
    echo "============================================="
    echo " Ablation: beta = $beta"
    echo "============================================="
    BETA="$beta" EXPERIMENT="adpo-ablation-beta${beta}" EPOCHS=2 \
        bash scripts/train_math.sh "$@" \
        || echo "[WARN] beta=$beta failed"
done
echo "Ablation sweep complete!"
