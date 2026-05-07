#!/usr/bin/env bash
# Thin wrapper around SPO's start_vllm_server.sh that forwards args.
# Usage: bash scripts/start_vllm_server.sh <model> <port> <seed> <swap_gb> [gpu_idx]

set -euo pipefail
INGPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${INGPO_ROOT}/spo/scripts/start_vllm_server.sh" "$@"
