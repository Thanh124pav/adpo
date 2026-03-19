#!/usr/bin/env bash
# ==========================================================================
# Run reasoning analysis across all (model x dataset) combinations.
#
# Directory structure:
#   reasoning_analysis/outputs/
#   ├── data/                          # prepared parquet (auto-downloaded)
#   ├── results/
#   │   ├── {model_key}/
#   │   │   ├── {dataset_key}.jsonl    # per-token analysis
#   │   │   └── ...
#   ├── visualizations/
#   │   ├── {model_key}/
#   │   │   ├── {dataset_key}/         # plots + html + eda/
#   │   │   │   ├── eda/
#   │   │   │   │   ├── boxplots.png
#   │   │   │   │   ├── histograms.png
#   │   │   │   │   ├── scatter_*.png
#   │   │   │   │   ├── summary.csv
#   │   │   │   │   └── eda_stats.json
#   │   │   │   ├── distribution_all_tokens.png
#   │   │   │   ├── neg_log_prob.html
#   │   │   │   ├── entropy.html
#   │   │   │   └── summary_stats.json
#   │   │   └── ...
#   │   └── ...
#   └── summary/                       # cross-experiment summary
#
# Usage:
#   bash reasoning_analysis/run_all_experiments.sh          # run everything
#   bash reasoning_analysis/run_all_experiments.sh --gpu 0  # specify GPU
#   bash reasoning_analysis/run_all_experiments.sh --skip-data  # skip data prep
#   bash reasoning_analysis/run_all_experiments.sh --viz-only   # only visualize
# ==========================================================================
set -euo pipefail

# ========================= CONFIGURATION ==================================

# --- Models ---------------------------------------------------------------
# Format: "KEY|PATH"
# KEY is used for directory naming, PATH is HuggingFace ID or local path.
MODELS=(
    "qwen3-0.6b|Qwen/Qwen3-0.6B"
    "qwen3-1.7b|Qwen/Qwen3-1.7B"
    "qwen3-4b|Qwen/Qwen3-4B"
    "llama3.2-1b|meta-llama/Llama-3.2-1B-Instruct"
    "llama3.2-3b|meta-llama/Llama-3.2-3B-Instruct"
    "qwen2.5-7b-math|Qwen/Qwen2.5-Math-7B-Instruct"
)

# --- Datasets -------------------------------------------------------------
# Format: "KEY|PREPARE_NAME"
# KEY = directory name, PREPARE_NAME = argument for prepare_datasets.py
DATASETS=(
    "math500|math500"
    "gsm8k|gsm8k_test"
    "big_math_rl|big_math_rl"
    "aops_instruct|aops_instruct"
)

# --- Inference params -----------------------------------------------------
TEMPERATURE=0.6
MAX_TOKENS=4096
TOP_P=0.95
N_SAMPLES=1
MAX_SAMPLES=-1            # -1 = all samples
TENSOR_PARALLEL_SIZE=1
TOP_LOGPROBS=20
BACKEND="vllm"

# --- Paths ----------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/processed"
OUTPUT_ROOT="${PROJECT_ROOT}/reasoning_analysis/outputs"
RESULTS_DIR="${OUTPUT_ROOT}/results"
VIZ_DIR="${OUTPUT_ROOT}/visualizations"
SUMMARY_DIR="${OUTPUT_ROOT}/summary"
LOG_DIR="${OUTPUT_ROOT}/logs"

# --- GPU ------------------------------------------------------------------
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

# ========================= CLI ARGS =======================================
SKIP_DATA=false
VIZ_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU_ID="$2"; shift 2 ;;
        --skip-data) SKIP_DATA=true; shift ;;
        --viz-only)  VIZ_ONLY=true; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --n-samples)   N_SAMPLES="$2"; shift 2 ;;
        --backend)     BACKEND="$2"; shift 2 ;;
        --tp)          TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ========================= HELPERS ========================================

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

elapsed() {
    local t=$1
    printf "%02d:%02d:%02d" $((t/3600)) $(( (t%3600)/60 )) $((t%60))
}

# ========================= STEP 0: PREPARE DATA ===========================

prepare_data() {
    log "========== STEP 0: Preparing datasets =========="
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r key prepare_name <<< "$entry"
        # Eval datasets go to data/processed/eval/, train to train/
        # Check both locations
        local parquet_eval="${DATA_DIR}/eval/${prepare_name}.parquet"
        local parquet_train="${DATA_DIR}/train/${prepare_name}.parquet"
        if [[ -f "$parquet_eval" ]] || [[ -f "$parquet_train" ]]; then
            log "  [SKIP] ${key}: already exists"
            continue
        fi
        log "  Preparing ${key} (${prepare_name}) ..."
        python "${PROJECT_ROOT}/data/prepare_datasets.py" \
            --dataset "$prepare_name" \
            --output_dir "$DATA_DIR" || {
            log "  [WARN] Failed to prepare ${key}, skipping"
        }
    done
    log "Data preparation done."
}

# Find the parquet file for a dataset key
find_parquet() {
    local prepare_name="$1"
    local parquet_eval="${DATA_DIR}/eval/${prepare_name}.parquet"
    local parquet_train="${DATA_DIR}/train/${prepare_name}.parquet"
    if [[ -f "$parquet_eval" ]]; then
        echo "$parquet_eval"
    elif [[ -f "$parquet_train" ]]; then
        echo "$parquet_train"
    else
        echo ""
    fi
}

# ========================= STEP 1: INFERENCE ==============================

run_inference() {
    log "========== STEP 1: Inference (evaluate.py) =========="
    local total=${#MODELS[@]}*${#DATASETS[@]}
    local count=0
    local failed=0

    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_key model_path <<< "$model_entry"

        for data_entry in "${DATASETS[@]}"; do
            IFS='|' read -r data_key prepare_name <<< "$data_entry"
            count=$((count + 1))

            local parquet
            parquet=$(find_parquet "$prepare_name")
            if [[ -z "$parquet" ]]; then
                log "  [${count}] SKIP ${model_key} x ${data_key}: parquet not found"
                failed=$((failed + 1))
                continue
            fi

            local out_path="${RESULTS_DIR}/${model_key}/${data_key}.jsonl"
            mkdir -p "$(dirname "$out_path")"
            mkdir -p "$LOG_DIR"

            # Skip if output already exists and is non-empty
            if [[ -f "$out_path" ]] && [[ -s "$out_path" ]]; then
                log "  [${count}] SKIP ${model_key} x ${data_key}: already done"
                continue
            fi

            local log_file="${LOG_DIR}/${model_key}__${data_key}.log"
            log "  [${count}] RUN  ${model_key} x ${data_key}"
            log "         model=${model_path}"
            log "         data=${parquet}"
            log "         out=${out_path}"

            local start_time
            start_time=$(date +%s)

            if python "${PROJECT_ROOT}/reasoning_analysis/evaluate.py" \
                --model_path "$model_path" \
                --dataset_path "$parquet" \
                --output_path "$out_path" \
                --max_samples "$MAX_SAMPLES" \
                --n_samples "$N_SAMPLES" \
                --temperature "$TEMPERATURE" \
                --max_tokens "$MAX_TOKENS" \
                --top_p "$TOP_P" \
                --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                --top_logprobs "$TOP_LOGPROBS" \
                --backend "$BACKEND" \
                2>&1 | tee "$log_file"; then

                local end_time
                end_time=$(date +%s)
                log "  [${count}] DONE ${model_key} x ${data_key} ($(elapsed $((end_time - start_time))))"
            else
                failed=$((failed + 1))
                log "  [${count}] FAIL ${model_key} x ${data_key} — see ${log_file}"
            fi
        done
    done

    log "Inference complete. Failed: ${failed}"
}

# ========================= STEP 2: VISUALIZATION + EDA ====================

run_visualization() {
    log "========== STEP 2: Visualization + EDA =========="

    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_key model_path <<< "$model_entry"

        for data_entry in "${DATASETS[@]}"; do
            IFS='|' read -r data_key prepare_name <<< "$data_entry"

            local result_path="${RESULTS_DIR}/${model_key}/${data_key}.jsonl"
            local viz_path="${VIZ_DIR}/${model_key}/${data_key}"

            if [[ ! -f "$result_path" ]] || [[ ! -s "$result_path" ]]; then
                log "  SKIP viz ${model_key} x ${data_key}: no results"
                continue
            fi

            log "  VIZ  ${model_key} x ${data_key} -> ${viz_path}"

            python "${PROJECT_ROOT}/reasoning_analysis/visualize.py" \
                --input_path "$result_path" \
                --output_dir "$viz_path" || {
                log "  [WARN] Visualization failed for ${model_key} x ${data_key}"
            }
        done
    done

    log "Visualization complete."
}

# ========================= STEP 3: CROSS-EXPERIMENT SUMMARY ===============

generate_summary() {
    log "========== STEP 3: Cross-experiment summary =========="
    mkdir -p "$SUMMARY_DIR"

    local csv="${SUMMARY_DIR}/all_experiments.csv"
    echo "model,dataset,accuracy,n_total,n_correct,nlp_mean,nlp_std,entropy_mean,entropy_std" > "$csv"

    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_key model_path <<< "$model_entry"

        for data_entry in "${DATASETS[@]}"; do
            IFS='|' read -r data_key prepare_name <<< "$data_entry"

            local eda_json="${VIZ_DIR}/${model_key}/${data_key}/eda/eda_stats.json"
            local stats_json="${VIZ_DIR}/${model_key}/${data_key}/summary_stats.json"

            if [[ -f "$eda_json" ]]; then
                python -c "
import json, sys
with open('$eda_json') as f:
    d = json.load(f)
a = d.get('all', {})
nlp = a.get('neg_log_prob_mean', {})
ent = a.get('entropy_mean', {})
print(f'$model_key,$data_key,{d.get(\"accuracy\",\"?\")},{d.get(\"n_total\",\"?\")},{d.get(\"n_correct\",\"?\")},{nlp.get(\"mean\",\"?\")},{nlp.get(\"std\",\"?\")},{ent.get(\"mean\",\"?\")},{ent.get(\"std\",\"?\")}')
" >> "$csv" 2>/dev/null || echo "${model_key},${data_key},?,?,?,?,?,?,?" >> "$csv"
            elif [[ -f "$stats_json" ]]; then
                python -c "
import json
with open('$stats_json') as f:
    d = json.load(f)
nlp = d.get('neg_log_prob', {})
ent = d.get('entropy', {})
print(f'$model_key,$data_key,,{d.get(\"total_tokens\",\"\")},,{nlp.get(\"mean\",\"?\")},{nlp.get(\"std\",\"?\")},{ent.get(\"mean\",\"?\")},{ent.get(\"std\",\"?\")}')
" >> "$csv" 2>/dev/null || echo "${model_key},${data_key},?,?,?,?,?,?,?" >> "$csv"
            else
                echo "${model_key},${data_key},?,?,?,?,?,?,?" >> "$csv"
            fi
        done
    done

    log "Summary saved to ${csv}"

    # Print summary table
    echo ""
    echo "==================== EXPERIMENT SUMMARY ===================="
    column -t -s',' "$csv" 2>/dev/null || cat "$csv"
    echo "============================================================"
}

# ========================= MAIN ===========================================

main() {
    log "============================================================"
    log "  Reasoning Analysis: Full Experiment Pipeline"
    log "  Models:   ${#MODELS[@]}"
    log "  Datasets: ${#DATASETS[@]}"
    log "  Total:    $((${#MODELS[@]} * ${#DATASETS[@]})) experiments"
    log "  GPU:      ${GPU_ID}"
    log "  Backend:  ${BACKEND}"
    log "============================================================"

    local global_start
    global_start=$(date +%s)

    if [[ "$VIZ_ONLY" == true ]]; then
        run_visualization
        generate_summary
    else
        if [[ "$SKIP_DATA" == false ]]; then
            prepare_data
        fi
        run_inference
        run_visualization
        generate_summary
    fi

    local global_end
    global_end=$(date +%s)
    log "============================================================"
    log "  ALL DONE in $(elapsed $((global_end - global_start)))"
    log "  Results:        ${RESULTS_DIR}/"
    log "  Visualizations: ${VIZ_DIR}/"
    log "  Summary:        ${SUMMARY_DIR}/"
    log "============================================================"
}

main
