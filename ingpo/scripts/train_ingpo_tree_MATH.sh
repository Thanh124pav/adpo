#!/usr/bin/env bash
# Train InGPO-tree on MATH with Qwen2.5-1.5B (mirror of SPO-tree-666 setup).
# Branch factor sweep is selected via INGPO_TREE={444,666,888} (default 666).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${INGPO_TREE}-qwen1.5b-math}"

CFGS="${INGPO_ROOT}/configs/polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
