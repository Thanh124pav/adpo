#!/usr/bin/env bash
# Shared environment setup for all InGPO scripts.
# Source this file at the top of every entry-point script.

set -euo pipefail

# Locate project paths.
INGPO_ROOT="${INGPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPO_ROOT="${SPO_ROOT:-${INGPO_ROOT}/spo}"
INGPO_SRC="${INGPO_ROOT}/ingpo_src"

if [[ ! -d "${SPO_ROOT}" ]]; then
  echo "[InGPO] SPO repo missing at ${SPO_ROOT}. Run scripts/setup.sh first." >&2
  exit 1
fi

# PYTHONPATH: SPO src first (so `treetune` is importable), then InGPO ext.
export PYTHONPATH="${SPO_ROOT}/src:${INGPO_SRC}:${PYTHONPATH:-}"

# Make jsonnet `import` resolve in this order:
#   1. InGPO configs/  -- our overrides
#   2. SPO  configs/   -- the inherited SPO library
#   3. SPO root        -- needed because some upstream SPO configs use cwd
#                         relative imports like 'configs/trainers/...' (SPO is
#                         normally launched with cwd=SPO_ROOT).
export APP_JSONNET_PATH="${INGPO_ROOT}/configs:${SPO_ROOT}/configs:${SPO_ROOT}"

# Standard SPO env vars.
export APP_SEED="${APP_SEED:-42}"
export MASTER_PORT="${MASTER_PORT:-$(python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"
export APP_DIRECTORY="${APP_DIRECTORY:-${INGPO_ROOT}/experiments}"
export APP_MINIMIZE_STORED_FILES="${APP_MINIMIZE_STORED_FILES:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-ingpo}"
mkdir -p "${APP_DIRECTORY}"

# Default GPU is index 0; override with INGPO_GPU=N.
INGPO_GPU="${INGPO_GPU:-0}"

# Ensure the SPO main entrypoint sees both config trees by chaining
# `--configs A,B,C`. We expose helpers for this below.

# Usage: ingpo_run "<exp_name>" "configs/foo.jsonnet,configs/bar.jsonnet" [extra deepspeed args]
ingpo_run() {
  local exp_name="$1"; shift
  local cfgs="$1"; shift
  WANDB_PROJECT="${WANDB_PROJECT}" \
  APP_EXPERIMENT_NAME="${exp_name}" \
  deepspeed --master_port "${MASTER_PORT}" --include "localhost:${INGPO_GPU}" \
    "${INGPO_SRC}/ingpo_main.py" \
    --configs "${cfgs},${SPO_ROOT}/configs/gpus/gpu_${INGPO_GPU}.jsonnet" \
    "$@" \
    run_iteration_loop
}

ingpo_eval() {
  local exp_name="$1"; shift
  local cfgs="$1"; shift
  local last_policy="$1"; shift
  WANDB_PROJECT="${WANDB_PROJECT}" \
  APP_EXPERIMENT_NAME="${exp_name}" \
  deepspeed --master_port "${MASTER_PORT}" --include "localhost:${INGPO_GPU}" \
    "${INGPO_SRC}/ingpo_main.py" \
    --configs "${cfgs},${SPO_ROOT}/configs/gpus/gpu_${INGPO_GPU}.jsonnet" \
    "$@" \
    evaluate --iteration 0 --last_policy_path "${last_policy}"
}
