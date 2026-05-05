#!/usr/bin/env bash
# ==============================================================================
# train_tpo_4gpu.sh — 4-GPU TPO training via Accelerate
#
# Each GPU runs one TPO worker:
#   - GPU 0 (process 0): builds TPO trees + runs PPO trainer (main process)
#   - GPU 1-3 (process 1-3): build TPO trees in parallel
#
# vLLM is started/stopped internally by TPOEpisodeGenerator at each iteration.
# Each process starts its own vLLM server on a different port.
#
# Usage:
#   cd /home/user/adpo
#   bash tpo/scripts/train_tpo_4gpu.sh [CONFIG] [SEED] [EXPERIMENT_DIR]
#
# Arguments:
#   CONFIG          Path to jsonnet config (default: tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet)
#   SEED            Random seed (default: 42)
#   EXPERIMENT_DIR  Output directory (default: experiments/tpo_4gpu)
# ==============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
CONFIG="${1:-tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet}"
SEED="${2:-42}"
EXPERIMENT_DIR="${3:-experiments/tpo_4gpu}"
NUM_GPUS="${NUM_GPUS:-4}"

# --------------------------------------------------------------------------
# Resolve paths
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SPO_DIR="${REPO_ROOT}/spo"
ACCELERATE_CONFIG="${REPO_ROOT}/tpo/configs/accelerate/default_config.yaml"

echo "======================================================"
echo "  TPO Training — ${NUM_GPUS} GPUs (Accelerate)"
echo "======================================================"
echo "  Config          : ${CONFIG}"
echo "  Seed            : ${SEED}"
echo "  Experiment dir  : ${EXPERIMENT_DIR}"
echo "  Accelerate cfg  : ${ACCELERATE_CONFIG}"
echo "======================================================"

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
export APP_SEED="${SEED}"
export PYTHONPATH="${REPO_ROOT}:${SPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------------------------
# Run from SPO directory
# --------------------------------------------------------------------------
cd "${SPO_DIR}"

if [[ -f "${ACCELERATE_CONFIG}" ]]; then
    ACCEL_ARGS="--config_file ${ACCELERATE_CONFIG}"
else
    # Fallback: pass GPU count directly
    ACCEL_ARGS="--num_processes ${NUM_GPUS}"
fi

accelerate launch \
    ${ACCEL_ARGS} \
    "${REPO_ROOT}/tpo/scripts/train_tpo.py" \
    "${REPO_ROOT}/${CONFIG}" \
    run_iteration \
    --result_dir "${REPO_ROOT}/${EXPERIMENT_DIR}"
