#!/bin/bash
# Generate reference solutions for all training datasets.
#
# Two strategies:
#   STRATEGY=teacher  -> large model solves freely (default)
#   STRATEGY=hint     -> training model given the answer, produces reasoning
#
# Usage:
#   bash scripts/generate_all_solutions.sh
#   STRATEGY=hint MODEL=Qwen/Qwen2.5-Math-7B bash scripts/generate_all_solutions.sh
#   NUM_SOLUTIONS=8 bash scripts/generate_all_solutions.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

STRATEGY=${STRATEGY:-"teacher"}
MODEL=${MODEL:-"Qwen/Qwen2.5-72B-Instruct"}
NUM_SOLUTIONS=${NUM_SOLUTIONS:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
TP_SIZE=${TP_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-256}
GPU_MEM=${GPU_MEM:-0.85}

echo "============================================="
echo " Generate Reference Solutions"
echo " Strategy: $STRATEGY"
echo " Model:    $MODEL"
echo " N sols:   $NUM_SOLUTIONS"
echo "============================================="

DATASETS=(
    "math"
    "gsm8k"
    "numina_math"
    "open_math_reasoning"
    "aops_instruct"
    "big_math_rl"
    "fine_math"
)

for ds in "${DATASETS[@]}"; do
    INPUT="data/processed/train/${ds}.parquet"
    OUTPUT="data/solutions/${ds}_solutions.jsonl"

    if [ ! -f "$INPUT" ]; then
        echo "[SKIP] $ds: $INPUT not found (run prepare_all_data.sh first)"
        continue
    fi

    echo ""
    echo ">>> Generating solutions for $ds..."
    python data/generate_solutions.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --strategy "$STRATEGY" \
        --num_solutions "$NUM_SOLUTIONS" \
        --temperature "$TEMPERATURE" \
        --tensor_parallel_size "$TP_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --gpu_memory_utilization "$GPU_MEM" \
        --append \
        || echo "[WARN] $ds failed"
done

echo ""
echo "============================================="
echo " Solution generation complete!"
echo " Output dir: data/solutions/"
echo "============================================="
