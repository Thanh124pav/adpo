#!/usr/bin/env bash
# Train InGPO-tree on MATH with DeepSeek-Distill-Qwen-1.5B by default.
# Branch factor sweep is selected via INGPO_TREE, e.g.
# {444,666,888,6666,66666,44444} (default 666).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
MODEL="${MODEL:-/workspace/storage-shared/models/DeepSeek-Distill-Qwen-1.5B}"
MODEL_CONFIG_ALIAS="${MODEL_CONFIG_ALIAS:-qwen1b}"
MODEL_SLUG="$(basename "${MODEL}")"
EXP_NAME="${APP_EXPERIMENT_NAME:-ingpo-tree-${INGPO_TREE}-${MODEL_SLUG}-math}"

if [[ "${MODEL}" == */* ]]; then
  CFGS="${INGPO_ROOT}/configs/polIter_${MODEL_CONFIG_ALIAS}_ingpo_tree_MATH.jsonnet"
  CFGS+=",$(ensure_model_path_config "${MODEL}")"
else
  CFGS="${INGPO_ROOT}/configs/polIter_${MODEL}_ingpo_tree_MATH.jsonnet"
fi
CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
