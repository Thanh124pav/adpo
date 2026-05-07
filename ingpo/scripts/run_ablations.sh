#!/usr/bin/env bash
# Run all ablations from PLAN.md §5 on MATH/Qwen2.5-1.5B with 6-6-6 trees.
# Caller can restrict via ABLATIONS="abl1 abl4" etc.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
ABLATIONS="${ABLATIONS:-abl1 abl2 abl3 abl4 abl5 abl6 abl7}"

BASE_CFG="${INGPO_ROOT}/configs/polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet"
TREE_CFG="${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"

run() {
  local exp="$1"
  local addon="$2"
  local cfgs="${BASE_CFG},${TREE_CFG},${addon}"
  APP_EXPERIMENT_NAME="${exp}" ingpo_run "${exp}" "${cfgs}"
}

for abl in ${ABLATIONS}; do
  case "${abl}" in
    abl1)
      run "abl1-K5-m50"   "${INGPO_ROOT}/configs/ablations/abl1_K5_m50.jsonnet"
      run "abl1-K20-m100" "${INGPO_ROOT}/configs/ablations/abl1_K20_m100.jsonnet"
      run "abl1-K10-m200" "${INGPO_ROOT}/configs/ablations/abl1_K10_m200.jsonnet"
      ;;
    abl2)
      run "abl2-eta-0p005" "${INGPO_ROOT}/configs/ablations/abl2_eta_0p005.jsonnet"
      run "abl2-eta-0p01"  "${INGPO_ROOT}/configs/ablations/abl2_eta_0p01.jsonnet"
      run "abl2-eta-0p02"  "${INGPO_ROOT}/configs/ablations/abl2_eta_0p02.jsonnet"
      run "abl2-eta-0p05"  "${INGPO_ROOT}/configs/ablations/abl2_eta_0p05.jsonnet"
      ;;
    abl3)
      run "abl3-share-parent" "${INGPO_ROOT}/configs/ablations/abl3_share_target_parent.jsonnet"
      run "abl3-share-root"   "${INGPO_ROOT}/configs/ablations/abl3_share_target_root.jsonnet"
      ;;
    abl4)
      run "abl4-share-only"  "${INGPO_ROOT}/configs/ablations/abl4_share_only.jsonnet"
      run "abl4-prune-only"  "${INGPO_ROOT}/configs/ablations/abl4_prune_only.jsonnet"
      run "abl4-neither"     "${INGPO_ROOT}/configs/ablations/abl4_neither.jsonnet"
      ;;
    abl5)
      run "abl5-y-per-dataset" "${INGPO_ROOT}/configs/ablations/abl5_y_per_dataset.jsonnet"
      ;;
    abl6)
      run "abl6-no-dkw" "${INGPO_ROOT}/configs/ablations/abl6_logp_vs_prob.jsonnet"
      ;;
    abl7)
      run "abl7-oracle-record" "${INGPO_ROOT}/configs/ablations/abl7_oracle_record.jsonnet"
      ;;
    *)
      echo "[ablations] Unknown: ${abl}" >&2
      exit 2
      ;;
  esac
done
