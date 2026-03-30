#!/bin/bash
# Verify attention reconstruction for supported model families.
#
# Usage:
#   bash reasoning_analysis/attention_analysis/run_verify.sh
#
# Override:
#   MODELS="Qwen/Qwen3-0.6B" NUM_SAMPLES=5 \
#       bash reasoning_analysis/attention_analysis/run_verify.sh

set -e

MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen2.5-0.5B}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
DEVICE="${DEVICE:-auto}"

echo "============================================================"
echo "  Attention Reconstruction Verification"
echo "============================================================"
echo "  Models:      $MODELS"
echo "  Samples:     $NUM_SAMPLES"
echo "  Device:      $DEVICE"
echo "============================================================"
echo ""

python reasoning_analysis/attention_analysis/verify.py \
    --model_path $MODELS \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE"
