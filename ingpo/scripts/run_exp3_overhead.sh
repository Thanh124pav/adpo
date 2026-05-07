#!/usr/bin/env bash
# Exp 3 (PLAN.md §4): Overhead measurement.
#   Trains InGPO on a small fixed slice and prints LP-scoring time, BST time,
#   total wallclock vs SPO on identical data. We piggy-back on wandb's
#   `timing/episode_generation/*` metrics already emitted by SPO and the
#   `ingpo/*` metrics our episode generator adds.
#
# Usage: bash scripts/run_exp3_overhead.sh [INGPO_TREE=666]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INGPO_TREE="${INGPO_TREE:-666}"
NUM_ITER="${NUM_ITER:-10}"

# Cap iterations for the cost run.
EXTRA_OVERRIDES="--debug False --override 'num_iterations=${NUM_ITER}'"

EXP_NAME="${APP_EXPERIMENT_NAME:-exp3-overhead-${INGPO_TREE}}"
APP_EXPERIMENT_NAME="${EXP_NAME}-spo" \
  INGPO_TREE="${INGPO_TREE}" bash "${INGPO_ROOT}/scripts/run_baseline.sh" spo_tree_MATH ${EXTRA_OVERRIDES}
APP_EXPERIMENT_NAME="${EXP_NAME}-ingpo" \
  INGPO_TREE="${INGPO_TREE}" bash "${INGPO_ROOT}/scripts/train_ingpo_tree_MATH.sh" ${EXTRA_OVERRIDES}
