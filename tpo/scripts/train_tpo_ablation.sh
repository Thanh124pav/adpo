#!/usr/bin/env bash
# ==============================================================================
# train_tpo_ablation.sh — Run TPO ablation variants
#
# Runs three configs sequentially (or pick one via VARIANT argument):
#   default      — entropy branching + P-degradation (main TPO)
#   deep_tree    — larger K=5, depth=12, adaptive tokens
#   fixed_branch — fixed branch_factor=3 (ablation: no entropy branching)
#
# Usage:
#   bash tpo/scripts/train_tpo_ablation.sh [VARIANT] [NUM_GPUS] [SEED]
#
# Examples:
#   bash tpo/scripts/train_tpo_ablation.sh                     # run all 3
#   bash tpo/scripts/train_tpo_ablation.sh fixed_branch 4 42   # one variant
# ==============================================================================

set -euo pipefail

VARIANT="${1:-all}"
NUM_GPUS="${2:-4}"
SEED="${3:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SPO_DIR="${REPO_ROOT}/spo"

export APP_SEED="${SEED}"
export PYTHONPATH="${REPO_ROOT}:${SPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

declare -A CONFIGS=(
    [default]="tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet"
    [deep_tree]="tpo/configs/polIter_qwen1_5b_tpo_MATH_deep_tree.jsonnet"
    [fixed_branch]="tpo/configs/polIter_qwen1_5b_tpo_MATH_fixed_branch.jsonnet"
)

run_variant() {
    local name="$1"
    local config="${CONFIGS[$name]}"
    local outdir="experiments/tpo_${name}_seed${SEED}"

    echo ""
    echo "======================================================"
    echo "  TPO Ablation: ${name}  (${NUM_GPUS} GPUs)"
    echo "  Config : ${config}"
    echo "  Output : ${outdir}"
    echo "======================================================"

    cd "${SPO_DIR}"

    if [[ "${NUM_GPUS}" -eq 1 ]]; then
        python "${REPO_ROOT}/tpo/scripts/train_tpo.py" \
            "${REPO_ROOT}/${config}" \
            run_iteration \
            --result_dir "${REPO_ROOT}/${outdir}"
    else
        ACCEL_CFG="${REPO_ROOT}/tpo/configs/accelerate/default_config.yaml"
        if [[ -f "${ACCEL_CFG}" ]]; then
            ACCEL_ARGS="--config_file ${ACCEL_CFG} --num_processes ${NUM_GPUS}"
        else
            ACCEL_ARGS="--num_processes ${NUM_GPUS}"
        fi
        accelerate launch ${ACCEL_ARGS} \
            "${REPO_ROOT}/tpo/scripts/train_tpo.py" \
            "${REPO_ROOT}/${config}" \
            run_iteration \
            --result_dir "${REPO_ROOT}/${outdir}"
    fi
}

if [[ "${VARIANT}" == "all" ]]; then
    for v in default fixed_branch deep_tree; do
        run_variant "${v}"
    done
else
    if [[ -z "${CONFIGS[$VARIANT]+_}" ]]; then
        echo "ERROR: Unknown variant '${VARIANT}'. Choose: default, deep_tree, fixed_branch, all"
        exit 1
    fi
    run_variant "${VARIANT}"
fi

echo ""
echo "Done."
