#!/usr/bin/env bash
# Sweep (K, m) — the InGPO scoring budget — to find the cheapest setting
# that still yields high prune/share precision. Reuses the abl1_*.jsonnet
# configs.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${INGPO_TREE:-666}}"
GRID="${GRID:-K1_m20 K5_m50 K10_m200 K20_m100 K20_m500}"
TAG="${EXP_TAG:-exp-K-m-sweep}"

BASE_CFG="${INGPO_ROOT}/configs/polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet"
TREE_CFG="$(ensure_tree_config "${TREE}")"

for cfg in ${GRID}; do
  cfgs="${BASE_CFG},${TREE_CFG},${INGPO_ROOT}/configs/ablations/abl1_${cfg}.jsonnet"
  APP_EXPERIMENT_NAME="${TAG}-${cfg}" ingpo_run "${TAG}-${cfg}" "${cfgs}" "$@"
done
