#!/usr/bin/env bash
# Train InGPO-tree on Point24 (24-game) with DeepSeek-R1-Distill-Qwen-1.5B.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${TREE}-qwen1b-point24}"
CFGS="${INGPO_ROOT}/configs/polIter_qwen1b_ingpo_tree_point24.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
