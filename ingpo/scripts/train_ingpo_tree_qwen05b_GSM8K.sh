#!/usr/bin/env bash
# Train InGPO-tree on GSM8K with Qwen-0.5B (smallest model — fast iteration).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${TREE}-qwen05b-gsm8k}"
CFGS="${INGPO_ROOT}/configs/polIter_qwen05b_ingpo_tree_GSM8K.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
