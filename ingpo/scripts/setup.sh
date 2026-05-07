#!/usr/bin/env bash
# One-time bootstrap for the InGPO project.
#   * verifies the cloned SPO repo
#   * installs SPO + InGPO Python deps
#   * downloads the same datasets SPO uses
#
# Usage: bash scripts/setup.sh

set -euo pipefail

INGPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPO_ROOT="${INGPO_ROOT}/spo"

if [[ ! -d "${SPO_ROOT}" ]]; then
  echo "[setup] Cloning SPO into ${SPO_ROOT}"
  git clone https://github.com/AIFrameResearch/SPO.git "${SPO_ROOT}"
fi

echo "[setup] Installing SPO Python dependencies"
pip install -r "${SPO_ROOT}/requirements.txt"

echo "[setup] Installing InGPO-specific dependencies"
pip install sortedcontainers httpx 'openai>=1.0' wandb

echo "[setup] Preparing datasets via SPO download script"
pushd "${SPO_ROOT}" > /dev/null
bash scripts/download_and_prepare_dataset.sh
popd > /dev/null

echo "[setup] Done."
