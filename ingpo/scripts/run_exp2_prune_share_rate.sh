#!/usr/bin/env bash
# Exp 2 (PLAN.md §4): Online prune/share rate per depth + advantage variance.
# Adds the abl7 oracle config so PRUNE/SHARE edges remain in the dataset and
# are emitted to wandb metrics for later post-hoc inspection.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
EXP_NAME="${APP_EXPERIMENT_NAME:-exp2-prune-share-${INGPO_TREE}-qwen1.5b-math}"

CFGS="${INGPO_ROOT}/configs/polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/ablations/abl7_oracle_record.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
