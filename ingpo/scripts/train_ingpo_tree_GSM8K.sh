#!/usr/bin/env bash
# Train InGPO-tree on GSM8K with Rho-1.1B-SFT.
# Tree shape via TREE=<digits> (INGPO_TREE also accepted).  See
# train_ingpo_tree_MATH.sh for the auto-generation rules.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${TREE}-rho1.1b-gsm8k}"
CFGS="${INGPO_ROOT}/configs/polIter_rho1bSft2_ingpo_tree_GSM8K.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
