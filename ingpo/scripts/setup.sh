#!/usr/bin/env bash
# One-time bootstrap for the unified treetune project.
#   * installs Python deps (treetune + InGPO core helpers)
#   * downloads the same datasets SPO uses
#
# Usage: bash scripts/setup.sh

set -euo pipefail

INGPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] Installing Python dependencies"
pip install -r "${INGPO_ROOT}/requirements.txt"
pip install sortedcontainers httpx 'openai>=1.0' wandb

echo "[setup] Preparing datasets"
pushd "${INGPO_ROOT}" > /dev/null
bash scripts/download_and_prepare_dataset.sh
popd > /dev/null

echo "[setup] Done."
