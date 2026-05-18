#!/usr/bin/env bash
# Train SPO-chain on MATH.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen1b}"
EXP_NAME="${APP_EXPERIMENT_NAME:-spo-chain-${MODEL}-math}"

CFGS="${INGPO_ROOT}/configs/polIter_${MODEL}_spo_chain_MATH.jsonnet"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
