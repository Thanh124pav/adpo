#!/usr/bin/env bash
# Train SPO-tree on MATH.  Tree shape via TREE={444,666,888,6666,66666,666666}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen1b}"
TREE="${TREE:-666}"
EXP_NAME="${APP_EXPERIMENT_NAME:-spo-tree-${TREE}-${MODEL}-math}"

CFGS="${INGPO_ROOT}/configs/polIter_${MODEL}_spo_tree_MATH.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${TREE}.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
