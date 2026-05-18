#!/usr/bin/env bash
# Train SPO-tree on MATH.
#
# Tree shape via TREE=<digits> (any shape works — ensure_tree_config
# auto-generates the overlay if no checked-in file matches).
#   TREE=666      depth 3 (default)
#   TREE=6666     depth 4, M=500
#   TREE=66666    depth 5, M=400
#   TREE=666666   depth 6, M=300
#   TREE=8888     depth 4, M=500 (auto-generated)
#   TREE=3456     depth 4, M=500 (auto-generated, mixed widths)
# Override M with TREE_M=<int>.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen1b}"
TREE="${TREE:-${INGPO_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-spo-tree-${TREE}-${MODEL}-math}"

CFGS="${INGPO_ROOT}/configs/polIter_${MODEL}_spo_tree_MATH.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"

ingpo_run "${EXP_NAME}" "${CFGS}" "$@"
