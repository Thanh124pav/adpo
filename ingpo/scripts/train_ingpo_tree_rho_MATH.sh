#!/usr/bin/env bash
# Train InGPO-tree on MATH with Rho-1.1B-SFT.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${TREE}-rho1.1b-math}"
CFGS="${INGPO_ROOT}/configs/polIter_rho1bSft2_ingpo_tree_MATH.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
