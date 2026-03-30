#!/bin/bash
# Verify attention reconstruction for supported model families.
#
# Usage:
#   # Quick test (small models only)
#   bash reasoning_analysis/attention_analysis/run_verify.sh
#
#   # Test specific models
#   MODELS="Qwen/Qwen3-0.6B" NUM_SAMPLES=5 \
#       bash reasoning_analysis/attention_analysis/run_verify.sh
#
#   # Test all supported families (needs more VRAM)
#   bash reasoning_analysis/attention_analysis/run_verify.sh --all
#
# Supported model families:
#   - Qwen2.5:   Qwen/Qwen2.5-0.5B (GQA, RoPE, sliding window)
#   - Qwen3:     Qwen/Qwen3-0.6B   (GQA, RoPE, QK-Norm)
#   - Llama3:    meta-llama/Llama-3.2-1B (GQA, RoPE)
#   - DeepSeek:  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 arch)

set -e

NUM_SAMPLES="${NUM_SAMPLES:-10}"
DEVICE="${DEVICE:-auto}"

if [ "$1" = "--all" ]; then
    MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
else
    MODELS="${MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen2.5-0.5B}"
fi

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
