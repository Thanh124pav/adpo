#!/bin/bash
# Full pipeline: Inference → Analysis → Attention Extraction → Visualization
#
# Usage:
#   bash reasoning_analysis/run_full_pipeline.sh
#
# Override defaults via environment variables:
#   MODEL=Qwen/Qwen2.5-14B-Instruct \
#   DATASET=data/aime2025.parquet \
#   MAX_SAMPLES=100 \
#   bash reasoning_analysis/run_full_pipeline.sh
#
# Prerequisites:
#   pip install torch transformers vllm datasets pandas pyarrow matplotlib numpy

set -e

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-data/processed/eval/gsm8k_test.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-"reasoning_analysis/outputs/hidden_states_analysis_$MODEL" }"
MAX_SAMPLES="${MAX_SAMPLES:-15}"         # Number of prompts (-1 = all)
N_SAMPLES="${N_SAMPLES:-4}"              # Responses per prompt
BACKEND="${BACKEND:-hf}"               # vllm or hf
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TOP_LOGPROBS="${TOP_LOGPROBS:-20}"
TP_SIZE="${TP_SIZE:-1}"                  # tensor_parallel_size for vLLM
DEVICE="${DEVICE:-auto}"                 # Device for HF forward pass
EXACT_ENTROPY="${EXACT_ENTROPY:-0}"      # Set 1 to enable exact entropy
ATTN_IMPL="${ATTN_IMPL:-eager}"  # flash_attention_2 (default) or eager
VIZ_ONLY="${VIZ_ONLY:-False}"
LAYERS="${LAYERS:-None}"
NO_PNG="${NO_PNG:-0}"
# Derived paths
ANALYSIS_FILE="$OUTPUT_DIR/analysis.jsonl"
INTERNALS_DIR="$OUTPUT_DIR/internals"
VIZ_DIR="$OUTPUT_DIR/visualizations"

# Optional flags
EXTRA_FLAGS=""
if [ "$EXACT_ENTROPY" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --exact_entropy"
fi

# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Reasoning Analysis: Full Pipeline"
echo "============================================================"
echo "  Model:        $MODEL"
echo "  Dataset:      $DATASET"
echo "  Backend:      $BACKEND"
echo "  Max samples:  $MAX_SAMPLES"
echo "  N samples:    $N_SAMPLES"
echo "  Max tokens:   $MAX_TOKENS"
echo "  Temperature:  $TEMPERATURE"
echo "  Output:       $OUTPUT_DIR"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 1: Prepare dataset (if using HuggingFace AIME2025)
# ---------------------------------------------------------------------------
if [ ! -f "$DATASET" ]; then
    echo "=== Step 0: Dataset not found, downloading AIME 2025 ==="
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
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\\\boxed{}.'},
                {'role': 'user', 'content': row.get('question', row.get('problem', ''))}
            ],
            'ground_truth': str(row.get('answer', '')),
        })

with open('${DATASET}', 'w') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f'Saved {len(records)} samples to ${DATASET}')
"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Inference + log_prob/entropy + attention extraction
# ---------------------------------------------------------------------------

if [ "$VIZ_ONLY" == "False" ]; then
    echo "=== Step 1: Inference + Hidden States Extraction ==="
    echo "  This runs vLLM generation, then HF forward pass for hidden states."
    echo "  (Attention matrices will be reconstructed at visualization time)"
    echo ""
    python reasoning_analysis/evaluate.py \
        --model_path "$MODEL" \
        --dataset_path "$DATASET" \
        --output_path "$ANALYSIS_FILE" \
        --backend "$BACKEND" \
        --max_samples "$MAX_SAMPLES" \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --top_logprobs "$TOP_LOGPROBS" \
        --tensor_parallel_size "$TP_SIZE" \
        --device "$DEVICE" \
        --extract_internals \
        --attn_impl "$ATTN_IMPL" \
        $EXTRA_FLAGS

    echo ""
    echo "  Analysis saved to: $ANALYSIS_FILE"
    echo "  Internals saved to: $INTERNALS_DIR/"
    echo ""
else
    echo ""
    echo "=== Skip Step 1 ===="
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Visualization
# ---------------------------------------------------------------------------
echo "=== Step 2: Visualization ==="
echo "  Generating: log_prob HTML, entropy HTML, statistical plots, attention heatmaps"
echo ""

VIZ_FLAGS=""
if [ "$NO_PNG" = "1" ]; then
    VIZ_FLAGS="$VIZ_FLAGS --no_png"
fi

python reasoning_analysis/visualize.py \
    --input_path "$ANALYSIS_FILE" \
    --internals_dir "$INTERNALS_DIR" \
    --model_path "$MODEL" \
    --attn_impl "$ATTN_IMPL" \
    --output_dir "$VIZ_DIR" \
    --layers $LAYERS \
    $VIZ_FLAGS

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── analysis.jsonl                  # per-token log_prob + entropy"
echo "  ├── internals/                      # attention matrices (.npz)"

if [ -d "$INTERNALS_DIR" ]; then
    NPZ_COUNT=$(ls "$INTERNALS_DIR"/*.npz 2>/dev/null | wc -l)
    echo "  │   └── ($NPZ_COUNT .npz files)"
fi

echo "  └── visualizations/"
echo "      ├── neg_log_prob.html           # token-level log_prob (50 samples)"
echo "      ├── entropy.html                # token-level entropy (50 samples)"
echo "      ├── distributions.png           # statistical distributions"
echo "      ├── per_response_stats.png      # per-response statistics"
echo "      ├── per_position_trends.png     # positional trends"
echo "      └── attention_heatmaps/         # attention heatmaps (50 samples)"
echo "          ├── sample_XXX_think_think_tokens.png"
echo "          ├── sample_XXX_think_think_sentences.png"
echo "          ├── sample_XXX_out_think_tokens.png"
echo "          └── sample_XXX_out_think_sentences.png"
echo ""

# File sizes
echo "File sizes:"
find "$OUTPUT_DIR" -type f -name "*.jsonl" -o -name "*.html" -o -name "*.png" 2>/dev/null | \
    while read f; do
        SIZE=$(du -h "$f" | cut -f1)
        echo "  $SIZE  $f"
    done
