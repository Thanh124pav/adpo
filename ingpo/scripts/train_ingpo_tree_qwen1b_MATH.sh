#!/usr/bin/env bash
# Train InGPO-tree on MATH with DeepSeek-R1-Distill-Qwen-1.5B (long-CoT).
# This is the "long-CoT" InGPO setup whose break-even depth was discussed in
# the analysis notes — set INGPO_TREE=4444 to actually realise it.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${INGPO_TREE}-qwen1b-math}"
CFGS="${INGPO_ROOT}/configs/polIter_qwen1b_ingpo_tree_MATH.jsonnet"
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
