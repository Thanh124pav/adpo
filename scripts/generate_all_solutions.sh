#!/bin/bash
# Generate reference solutions for all training datasets.
#
# Two modes:
#   USE_API=0 (default) -> local vLLM generation (generate_solutions.py)
#   USE_API=1           -> API-based generation (generate_solutions_api.py)
#
# Strategies:
#   Local:  teacher | hint
#   API:    hint | free | both
#
# Usage:
#   # Local vLLM (default)
#   bash scripts/generate_all_solutions.sh
#   STRATEGY=hint MODEL=Qwen/Qwen2.5-Math-7B bash scripts/generate_all_solutions.sh
#
#   # API-based
#   USE_API=1 bash scripts/generate_all_solutions.sh
#   USE_API=1 STRATEGY=both ENDPOINT=http://10.254.138.189:8104 bash scripts/generate_all_solutions.sh
#   USE_API=1 STRATEGY=free NUM_SOLUTIONS=8 bash scripts/generate_all_solutions.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

USE_API=${USE_API:-0}
STRATEGY=${STRATEGY:-"teacher"}
MODEL=${MODEL:-"Qwen/Qwen2.5-72B-Instruct"}
NUM_SOLUTIONS=${NUM_SOLUTIONS:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
BATCH_SIZE=${BATCH_SIZE:-256}

# Local vLLM options
TP_SIZE=${TP_SIZE:-4}
GPU_MEM=${GPU_MEM:-0.85}

# API options
ENDPOINT=${ENDPOINT:-"http://10.254.138.189:8104"}
MAX_CONCURRENT=${MAX_CONCURRENT:-64}
TIMEOUT=${TIMEOUT:-120}

if [ "$USE_API" -eq 1 ]; then
    # Default strategy for API mode is "hint" (not "teacher")
    STRATEGY=${STRATEGY:-"hint"}
    echo "============================================="
    echo " Generate Reference Solutions (API)"
    echo " Endpoint: $ENDPOINT"
    echo " Model:    $MODEL"
    echo " Strategy: $STRATEGY"
    echo " N sols:   $NUM_SOLUTIONS"
    echo " Concurrent: $MAX_CONCURRENT"
    echo "============================================="
else
    echo "============================================="
    echo " Generate Reference Solutions (Local vLLM)"
    echo " Model:    $MODEL"
    echo " Strategy: $STRATEGY"
    echo " N sols:   $NUM_SOLUTIONS"
    echo "============================================="
fi

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

    if [ "$USE_API" -eq 1 ]; then
        python data/generate_solutions_api.py \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --endpoint "$ENDPOINT" \
            --model "$MODEL" \
            --strategy "$STRATEGY" \
            --num_solutions "$NUM_SOLUTIONS" \
            --temperature "$TEMPERATURE" \
            --max_concurrent "$MAX_CONCURRENT" \
            --timeout "$TIMEOUT" \
            --batch_size "$BATCH_SIZE" \
            --append \
            || echo "[WARN] $ds failed"
    else
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
    fi
done

echo ""
echo "============================================="
echo " Solution generation complete!"
echo " Output dir: data/solutions/"
echo "============================================="
