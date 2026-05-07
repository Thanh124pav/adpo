#!/usr/bin/env bash
# Train InGPO-tree on Point24 (24-game) with DeepSeek-R1-Distill-Qwen-1.5B.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${INGPO_TREE}-qwen1b-point24}"
CFGS="${INGPO_ROOT}/configs/polIter_qwen1b_ingpo_tree_point24.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
