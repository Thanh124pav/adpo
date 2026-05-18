#!/usr/bin/env bash
# Train DPO (positive variant) on MATH.  Default model: rho1bSft2.
# Override base with MODEL={rho1bSft2,deepseekSft2}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
EXP_NAME="${APP_EXPERIMENT_NAME:-dpo-${MODEL}-math}"

CFGS="${INGPO_ROOT}/configs/polIter_${MODEL}_dpo_positive_MATH.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
