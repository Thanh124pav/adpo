#!/usr/bin/env bash
# ==============================================================================
# train_tpo_1gpu.sh — Single-GPU TPO training (debug / quick experiments)
#
# Mirrors SPO's single-GPU training pattern.  vLLM is started and stopped
# internally by TPOEpisodeGenerator at each iteration — no separate server
# process is needed.
#
# Usage:
#   cd /home/user/adpo
#   bash tpo/scripts/train_tpo_1gpu.sh [CONFIG] [SEED] [EXPERIMENT_DIR]
#
# Arguments:
#   CONFIG          Path to jsonnet config (default: tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet)
#   SEED            Random seed (default: 42)
#   EXPERIMENT_DIR  Output directory (default: experiments/tpo_1gpu)
#
# Examples:
#   bash tpo/scripts/train_tpo_1gpu.sh
#   bash tpo/scripts/train_tpo_1gpu.sh tpo/configs/polIter_qwen1_5b_tpo_MATH_deep_tree.jsonnet 0
#   bash tpo/scripts/train_tpo_1gpu.sh tpo/configs/polIter_qwen1_5b_tpo_MATH_fixed_branch.jsonnet 123
# ==============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
CONFIG="${1:-tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet}"
SEED="${2:-42}"
EXPERIMENT_DIR="${3:-experiments/tpo_1gpu}"

# --------------------------------------------------------------------------
# Resolve repo root regardless of where the script is called from
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # /home/user/adpo
SPO_DIR="${REPO_ROOT}/spo"

echo "======================================================"
echo "  TPO Training — Single GPU"
echo "======================================================"
echo "  Config        : ${CONFIG}"
echo "  Seed          : ${SEED}"
echo "  Experiment dir: ${EXPERIMENT_DIR}"
echo "  Repo root     : ${REPO_ROOT}"
echo "======================================================"

# --------------------------------------------------------------------------
# Environment (mirrors SPO's env setup)
# --------------------------------------------------------------------------
export APP_SEED="${SEED}"
export PYTHONPATH="${REPO_ROOT}:${SPO_DIR}/src:${PYTHONPATH:-}"

# Use GPU 0 by default; override with CUDA_VISIBLE_DEVICES before calling
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# HuggingFace cache — set to a fast drive if available
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# Prevent tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------------------------
# Run from SPO directory (treetune.main expects to find configs relative to it)
# --------------------------------------------------------------------------
cd "${SPO_DIR}"

python "${REPO_ROOT}/tpo/scripts/train_tpo.py" \
    "${REPO_ROOT}/${CONFIG}" \
    run_iteration \
    --result_dir "${REPO_ROOT}/${EXPERIMENT_DIR}"
