#!/usr/bin/env bash
# Run a paper baseline (SPO-tree, PPO/GRPO, or RFT/ReSTEM).
# Usage:
#   bash scripts/run_baseline.sh spo_tree_MATH
#   bash scripts/run_baseline.sh ppo_MATH
#   bash scripts/run_baseline.sh rft_MATH
#   bash scripts/run_baseline.sh spo_tree_GSM8K [INGPO_TREE=666]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NAME="${1:?Usage: run_baseline.sh <baseline_name>}"
shift || true

CFG="${INGPO_ROOT}/configs/baselines/${NAME}.jsonnet"
[[ -f "${CFG}" ]] || { echo "Unknown baseline: ${NAME}"; exit 2; }

CFGS="${CFG}"
if [[ "${NAME}" == *spo_tree* && "${INGPO_TREE:-}" != "" ]]; then
  CFGS+=",${INGPO_ROOT}/configs/episode_generators/branch_factor_${INGPO_TREE}.jsonnet"
fi

EXP_NAME="${APP_EXPERIMENT_NAME:-baseline-${NAME}}"
ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
