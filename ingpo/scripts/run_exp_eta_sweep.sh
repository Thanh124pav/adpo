#!/usr/bin/env bash
# Sweep eta (the SHARE/PRUNE threshold) at fixed K=10, m=100 to plot the
# soundness vs. throughput trade-off.
#
# Reuses the abl2_eta_*.jsonnet configs (override Lemma 2.4 with explicit eta).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
ETAS="${ETAS:-0p005 0p01 0p02 0p05 0p10 0p20}"
TAG="${EXP_TAG:-exp-eta-sweep}"

BASE_CFG="${INGPO_ROOT}/configs/polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet"
TREE_CFG="${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"

for eta in ${ETAS}; do
  cfgs="${BASE_CFG},${TREE_CFG},${INGPO_ROOT}/configs/ablations/abl2_eta_${eta}.jsonnet"
  APP_EXPERIMENT_NAME="${TAG}-eta-${eta}" ingpo_run "${TAG}-eta-${eta}" "${cfgs}" "$@"
done
