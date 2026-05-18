#!/usr/bin/env bash
# Shared environment setup for all training scripts in this repo.
# Source this file at the top of every entry-point script.

set -euo pipefail

# Locate project paths.
INGPO_ROOT="${INGPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# PYTHONPATH: project root so `treetune` and `guidance` packages resolve.
export PYTHONPATH="${INGPO_ROOT}:${PYTHONPATH:-}"

# Jsonnet `import` resolves under the unified configs/ tree.
export APP_JSONNET_PATH="${INGPO_ROOT}/configs:${INGPO_ROOT}"

# Standard env vars (kept compatible with upstream SPO conventions).
export APP_SEED="${APP_SEED:-42}"
export MASTER_PORT="${MASTER_PORT:-$(python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"
export APP_DIRECTORY="${APP_DIRECTORY:-${INGPO_ROOT}/experiments}"
export APP_MINIMIZE_STORED_FILES="${APP_MINIMIZE_STORED_FILES:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-treetune}"
mkdir -p "${APP_DIRECTORY}"

# Default GPU is index 0; override with INGPO_GPU=N.
INGPO_GPU="${INGPO_GPU:-0}"

# Helpers ---------------------------------------------------------------------
# Run training with `--configs A,B,C`.  Algorithm selection is purely from the
# chained configs; the unified entry point is `treetune.main` (InGPO is
# auto-registered when `treetune` is imported).
#
# Usage: ingpo_run "<exp_name>" "configs/foo.jsonnet,configs/bar.jsonnet" [extra args]
ingpo_run() {
  local exp_name="$1"; shift
  local cfgs="$1"; shift
  WANDB_PROJECT="${WANDB_PROJECT}" \
  APP_EXPERIMENT_NAME="${exp_name}" \
  deepspeed --master_port "${MASTER_PORT}" --include "localhost:${INGPO_GPU}" \
    -m treetune.main \
    --configs "${cfgs},${INGPO_ROOT}/configs/gpus/gpu_${INGPO_GPU}.jsonnet" \
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
    -m treetune.main \
    --configs "${cfgs},${INGPO_ROOT}/configs/gpus/gpu_${INGPO_GPU}.jsonnet" \
    "$@" \
    evaluate --iteration 0 --last_policy_path "${last_policy}"
}
