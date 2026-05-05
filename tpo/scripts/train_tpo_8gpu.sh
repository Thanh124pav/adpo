#!/usr/bin/env bash
# ==============================================================================
# train_tpo_8gpu.sh — 8-GPU TPO training via Accelerate (full production run)
#
# Recommended setup:
#   - 8x A100/H100 80GB GPUs
#   - vllm_gpu_memory_utilization: 0.4 in config (leaves 60% for PPO training)
#   - ZeRO-0 deepspeed (config default: actor on each GPU independently)
#
# Usage:
#   cd /home/user/adpo
#   bash tpo/scripts/train_tpo_8gpu.sh [CONFIG] [SEED] [EXPERIMENT_DIR]
# ==============================================================================

set -euo pipefail

CONFIG="${1:-tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet}"
SEED="${2:-42}"
EXPERIMENT_DIR="${3:-experiments/tpo_8gpu}"
NUM_GPUS="${NUM_GPUS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SPO_DIR="${REPO_ROOT}/spo"
ACCELERATE_CONFIG="${REPO_ROOT}/tpo/configs/accelerate/default_config.yaml"

echo "======================================================"
echo "  TPO Training — ${NUM_GPUS} GPUs (Accelerate)"
echo "======================================================"
echo "  Config         : ${CONFIG}"
echo "  Seed           : ${SEED}"
echo "  Experiment dir : ${EXPERIMENT_DIR}"
echo "======================================================"

export APP_SEED="${SEED}"
export PYTHONPATH="${REPO_ROOT}:${SPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "${SPO_DIR}"

if [[ -f "${ACCELERATE_CONFIG}" ]]; then
    ACCEL_ARGS="--config_file ${ACCELERATE_CONFIG}"
else
    ACCEL_ARGS="--num_processes ${NUM_GPUS}"
fi

accelerate launch \
    ${ACCEL_ARGS} \
    "${REPO_ROOT}/tpo/scripts/train_tpo.py" \
    "${REPO_ROOT}/${CONFIG}" \
    run_iteration \
    --result_dir "${REPO_ROOT}/${EXPERIMENT_DIR}"
