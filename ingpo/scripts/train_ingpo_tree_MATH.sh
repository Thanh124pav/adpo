#!/usr/bin/env bash
# Train InGPO-tree on MATH with DeepSeek-R1-Distill-Qwen-1.5B by default.
# Branch factor sweep is selected via INGPO_TREE={444,666,888} (default 666).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
MODEL="${MODEL:-qwen1b}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${INGPO_TREE}-${MODEL}-math}"

CFGS="${INGPO_ROOT}/configs/polIter_${MODEL}_ingpo_tree_MATH.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
