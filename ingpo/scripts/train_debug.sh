#!/usr/bin/env bash
# Tiny smoke-test run: 2 iterations, depth-2 tree W=2, m=8.
# Verifies that the InGPO + SPO stack actually starts training end-to-end
# without burning GPU hours. Logs go to experiments/ingpo-debug-*.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-debug-$(date +%H%M%S)}"
CFGS="${INGPO_ROOT}/configs/polIter_qwen05b_ingpo_tree_GSM8K.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/debug.jsonnet"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
