#!/bin/bash
# =============================================================================
# Full Data Pipeline: Download → Prepare → Preview
#
# Usage:
#   # Full pipeline (all datasets)
#   bash scripts/data_pipeline.sh
#
#   # Specific dataset
#   bash scripts/data_pipeline.sh --dataset math
#
#   # Only hard problems
#   bash scripts/data_pipeline.sh --dataset math --level "Level 4,Level 5"
#
#   # Train only / eval only
#   bash scripts/data_pipeline.sh --train-only
#   bash scripts/data_pipeline.sh --eval-only
#
#   # Skip preview step
#   bash scripts/data_pipeline.sh --no-preview
#
#   # Custom output directory
#   OUTPUT_DIR=data/custom bash scripts/data_pipeline.sh
#
#   # Preview samples count
#   PREVIEW_SAMPLES=10 bash scripts/data_pipeline.sh --dataset math
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
PREVIEW_SAMPLES="${PREVIEW_SAMPLES:-3}"
LOG_DIR="${LOG_DIR:-logs/data_pipeline}"

TRAIN_DATASETS=(gsm8k math numina_math open_math_reasoning aops_instruct big_math_rl fine_math)
EVAL_DATASETS=(math500 gsm8k_test amc_2023 aime_2024 aime_2025 olympiad_bench minerva_math omni_math hmmt brumo cmimc)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
DATASET=""
LEVEL=""
MODE="all"         # all | train-only | eval-only | single
DO_PREVIEW=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2"; MODE="single"; shift 2 ;;
        --level)        LEVEL="$2"; shift 2 ;;
        --train-only)   MODE="train-only"; shift ;;
        --eval-only)    MODE="eval-only"; shift ;;
        --no-preview)   DO_PREVIEW=false; shift ;;
        --help|-h)
            head -28 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
RESET="\033[0m"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log_step() {
    echo -e "\n${BOLD}${CYAN}[$(timestamp)] STEP: $1${RESET}"
}

log_info() {
    echo -e "${GREEN}  [OK]${RESET} $1"
}

log_warn() {
    echo -e "${YELLOW}  [WARN]${RESET} $1"
}

log_fail() {
    echo -e "${RED}  [FAIL]${RESET} $1"
}

mkdir -p "$LOG_DIR"

# Track results
SUCCEEDED=()
FAILED=()
SKIPPED=()

# ---------------------------------------------------------------------------
# Step 1: Check dependencies
# ---------------------------------------------------------------------------
log_step "1/3  Checking dependencies"

MISSING=()
for pkg in datasets pandas pyarrow numpy; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING+=("$pkg")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    log_warn "Missing packages: ${MISSING[*]}"
    echo "  Installing..."
    pip install "${MISSING[@]}" -q
    log_info "Installed: ${MISSING[*]}"
else
    log_info "All dependencies available"
fi

# ---------------------------------------------------------------------------
# Step 2: Download & Prepare
# ---------------------------------------------------------------------------
log_step "2/3  Download & Prepare datasets → ${OUTPUT_DIR}"

LEVEL_ARG=""
if [[ -n "$LEVEL" ]]; then
    LEVEL_ARG="--level $LEVEL"
    echo -e "  Level filter: ${BOLD}${LEVEL}${RESET}"
fi

prepare_dataset() {
    local ds="$1"
    local split="${2:-train}"
    local log_file="${LOG_DIR}/${ds}.log"

    echo -ne "  Preparing ${BOLD}${ds}${RESET} (split=${split})..."

    if python3 data/prepare_datasets.py \
        --dataset "$ds" \
        --split "$split" \
        --output_dir "$OUTPUT_DIR" \
        $LEVEL_ARG \
        > "$log_file" 2>&1; then

        # Extract row count from log
        local count
        count=$(grep -oP 'Saved \K[0-9]+' "$log_file" | tail -1)
        log_info "${ds}: ${count:-?} rows"
        SUCCEEDED+=("$ds")
    else
        log_fail "${ds} (see ${log_file})"
        FAILED+=("$ds")
    fi
}

case "$MODE" in
    all)
        echo -e "\n  ${BOLD}--- Training datasets ---${RESET}"
        for ds in "${TRAIN_DATASETS[@]}"; do
            prepare_dataset "$ds" "train"
        done

        echo -e "\n  ${BOLD}--- Evaluation datasets ---${RESET}"
        for ds in "${EVAL_DATASETS[@]}"; do
            prepare_dataset "$ds" "test"
        done
        ;;

    train-only)
        echo -e "\n  ${BOLD}--- Training datasets ---${RESET}"
        for ds in "${TRAIN_DATASETS[@]}"; do
            prepare_dataset "$ds" "train"
        done
        ;;

    eval-only)
        echo -e "\n  ${BOLD}--- Evaluation datasets ---${RESET}"
        for ds in "${EVAL_DATASETS[@]}"; do
            prepare_dataset "$ds" "test"
        done
        ;;

    single)
        # Determine split: eval datasets default to test
        SPLIT="train"
        for eds in "${EVAL_DATASETS[@]}"; do
            if [[ "$eds" == "$DATASET" ]]; then
                SPLIT="test"
                break
            fi
        done
        prepare_dataset "$DATASET" "$SPLIT"
        ;;
esac

# ---------------------------------------------------------------------------
# Step 2.5: Summary table
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  ┌─────────────────────────────────────────────────┐${RESET}"
echo -e "${BOLD}  │  Preparation Summary                            │${RESET}"
echo -e "${BOLD}  ├─────────────────────────────────────────────────┤${RESET}"

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo -e "  │  ${GREEN}Succeeded:${RESET} ${#SUCCEEDED[@]} datasets"
    for ds in "${SUCCEEDED[@]}"; do
        echo -e "  │    ${GREEN}+${RESET} ${ds}"
    done
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "  │  ${RED}Failed:${RESET}    ${#FAILED[@]} datasets"
    for ds in "${FAILED[@]}"; do
        echo -e "  │    ${RED}x${RESET} ${ds}"
    done
fi
echo -e "${BOLD}  └─────────────────────────────────────────────────┘${RESET}"

# ---------------------------------------------------------------------------
# Step 3: Preview
# ---------------------------------------------------------------------------
if [[ "$DO_PREVIEW" == true ]] && [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    log_step "3/3  Preview processed datasets"

    if [[ "$MODE" == "single" ]]; then
        # Find the generated parquet file
        PARQUET_FILE=$(find "$OUTPUT_DIR" -name "${DATASET}*.parquet" -type f 2>/dev/null | head -1)
        if [[ -n "$PARQUET_FILE" ]]; then
            python3 scripts/preview_dataset.py "$PARQUET_FILE" \
                -k "$PREVIEW_SAMPLES" \
                ${LEVEL:+--level "$LEVEL"}
        fi
    else
        # Preview directory with stats-only for large batches, samples for small
        if [[ ${#SUCCEEDED[@]} -le 3 ]]; then
            python3 scripts/preview_dataset.py "$OUTPUT_DIR" \
                -k "$PREVIEW_SAMPLES" \
                ${LEVEL:+--level "$LEVEL"}
        else
            # Too many datasets — show stats summary only, then sample from first few
            echo ""
            echo "  Showing stats for all datasets, samples for first 3..."
            echo ""

            # Stats-only for all
            python3 scripts/preview_dataset.py "$OUTPUT_DIR" \
                --stats-only \
                ${LEVEL:+--level "$LEVEL"}

            # Samples for first 3
            COUNT=0
            for ds in "${SUCCEEDED[@]}"; do
                if [[ $COUNT -ge 3 ]]; then break; fi
                PARQUET_FILE=$(find "$OUTPUT_DIR" -name "${ds}*.parquet" -type f 2>/dev/null | head -1)
                if [[ -n "$PARQUET_FILE" ]]; then
                    python3 scripts/preview_dataset.py "$PARQUET_FILE" \
                        -k "$PREVIEW_SAMPLES" \
                        ${LEVEL:+--level "$LEVEL"}
                    COUNT=$((COUNT + 1))
                fi
            done
        fi
    fi
else
    if [[ "$DO_PREVIEW" == true ]]; then
        log_warn "Skipping preview (no datasets succeeded)"
    else
        echo -e "\n  Preview skipped (--no-preview)"
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}Pipeline complete!${RESET}"
echo "  Output:  ${OUTPUT_DIR}/"
echo "  Logs:    ${LOG_DIR}/"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "  ${YELLOW}Check failed dataset logs for details.${RESET}"
    exit 1
fi
