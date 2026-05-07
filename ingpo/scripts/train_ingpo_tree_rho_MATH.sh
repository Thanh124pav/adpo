#!/usr/bin/env bash
# Train InGPO-tree on MATH with Rho-1.1B-SFT.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${INGPO_TREE}-rho1.1b-math}"
CFGS="${INGPO_ROOT}/configs/polIter_rho1bSft2_ingpo_tree_MATH.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
