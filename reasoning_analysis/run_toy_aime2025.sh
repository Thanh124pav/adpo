#!/bin/bash
# Toy experiment: Qwen3-0.6B on 5 AIME 2025 samples
# Usage: bash reasoning_analysis/run_toy_aime2025.sh
#
# Prerequisites:
#   pip install torch transformers vllm datasets pandas pyarrow

set -e

MODEL="Qwen/Qwen3-0.6B"
MAX_SAMPLES=5
OUTPUT_DIR="reasoning_analysis/outputs/toy_aime2025"
DATA_FILE="$OUTPUT_DIR/aime2025_5samples.json"

mkdir -p "$OUTPUT_DIR"

# Step 1: Prepare a small AIME 2025 dataset (5 samples)
echo "=== Step 1: Preparing AIME 2025 data (${MAX_SAMPLES} samples) ==="
python -c "
import json
from datasets import load_dataset

records = []
for config in ('AIME2025-I', 'AIME2025-II'):
    ds = load_dataset('opencompass/AIME2025', config, split='test')
    for row in ds:
        records.append({
            'data_source': 'aime_2025',
            'prompt': [
                {'role': 'system', 'content': 'Please reason step by step, each reasoning phase should ended with \\\".\\\", and put your final answer within \\\\boxed{}.'},
                {'role': 'user', 'content': row.get('question', row.get('problem', ''))}
            ],
            'ground_truth': str(row.get('answer', '')),
        })
        if len(records) >= ${MAX_SAMPLES}:
            break
    if len(records) >= ${MAX_SAMPLES}:
        break

with open('${DATA_FILE}', 'w') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f'Saved {len(records)} samples to ${DATA_FILE}')
"

# Step 2: Run reasoning analysis with HF backend (no vLLM needed for small model)
echo ""
echo "=== Step 2: Running reasoning analysis with ${MODEL} ==="
python reasoning_analysis/evaluate.py \
    --model_path "$MODEL" \
    --dataset_path "$DATA_FILE" \
    --output_path "$OUTPUT_DIR/analysis.jsonl" \
    --max_samples "$MAX_SAMPLES" \
    --n_samples 1 \
    --temperature 0.6 \
    --max_tokens 2048 \
    --backend hf \
    --extract_internals

# Step 3: Visualize results
echo ""
echo "=== Step 3: Generating visualizations ==="
python reasoning_analysis/visualize.py \
    --input_path "$OUTPUT_DIR/analysis.jsonl" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Done! ==="
echo "Output files in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "Internals (attention & hidden states):"
ls -lh "$OUTPUT_DIR/internals/" 2>/dev/null || echo "  (no internals extracted)"
