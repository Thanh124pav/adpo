#!/usr/bin/env bash
# run_local.sh — run mc_analysis on a LOCAL machine with ~4 GB VRAM
#
# Uses HuggingFace backend (no vLLM server needed) with 4-bit quantization
# so the model fits in 4 GB VRAM.
#
# Requirements:
#   pip install torch transformers bitsandbytes accelerate numpy
#
# Usage:
#   bash run_local.sh [deepseek|rho] [4-4-4|6-6-6|8-8-8] [question_idx]
#
# Examples:
#   bash run_local.sh                     # deepseek, all trees, all questions
#   bash run_local.sh deepseek 4-4-4 0   # deepseek, 4-4-4 tree, question 0 only
#   bash run_local.sh rho 4-4-4           # rho-math, 4-4-4 tree, all questions
#
# Notes:
#   - 4-4-4 tree (85 nodes)  ≈ 5–15 min on a single consumer GPU
#   - 6-6-6 tree (259 nodes) ≈ 20–60 min
#   - 8-8-8 tree (585 nodes) ≈ 1–3 hours  ← only attempt if time permits
#   - All results saved as JSON + HTML under ./results/local/<model>/

set -euo pipefail

# ── args ─────────────────────────────────────────────────────────────────────
MODEL_KEY="${1:-deepseek}"
TREE="${2:-4-4-4}"           # single tree; run all three by looping (see below)
QUESTION_IDX="${3:-}"        # empty = all questions

# ── model map ────────────────────────────────────────────────────────────────
case "$MODEL_KEY" in
  deepseek) MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ;;
  rho)      MODEL="microsoft/rho-math-1.1b-v0.1" ;;
  *)        echo "Unknown model key '$MODEL_KEY'. Use: deepseek | rho"; exit 1 ;;
esac

SAVE_DIR="./results/local/${MODEL_KEY}"

echo "============================================================"
echo "  Backend  : HuggingFace (local, 4-bit quantization)"
echo "  Model    : $MODEL"
echo "  Tree     : $TREE"
echo "  Save dir : $SAVE_DIR"
echo "============================================================"

# ── check bitsandbytes ───────────────────────────────────────────────────────
python3 -c "import bitsandbytes" 2>/dev/null || {
  echo "[warn] bitsandbytes not found — installing..."
  pip install bitsandbytes accelerate --quiet
}

# ── run via inline Python ────────────────────────────────────────────────────
# (avoids a separate driver script; all config is right here)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 - <<PYEOF
import sys, json, time, asyncio
from pathlib import Path

sys.path.insert(0, str(Path("$SCRIPT_DIR").resolve().parents[1]))

from mc_analysis import HFBackend, analyse_hf, print_tree, visualise_html

SAMPLE_QUESTIONS = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning "
            "and bakes muffins for her friends every day with 4. She sells the remainder "
            "at the farmers' market daily for \$2 per fresh duck egg. "
            "How much in dollars does she make every day at the farmers' market?"
        ),
        "gold_answer": "18",
        "source": "GSM8K",
    },
    {
        "question": "Find the largest prime factor of \$9879\$.",
        "gold_answer": "37",
        "source": "MATH",
    },
    {
        "question": (
            "A rectangular garden has a perimeter of 54 meters. "
            "The length is 3 meters more than twice the width. "
            "What is the area of the garden in square meters?"
        ),
        "gold_answer": "168",
        "source": "custom",
    },
]

TREE_CONFIGS = {
    "4-4-4": {"branch_factor": 4, "max_depth": 3},
    "6-6-6": {"branch_factor": 6, "max_depth": 3},
    "8-8-8": {"branch_factor": 8, "max_depth": 3},
}

model_name  = "$MODEL"
tree_label  = "$TREE"
save_dir    = Path("$SAVE_DIR")
q_idx_str   = "$QUESTION_IDX"

tree_cfg    = TREE_CONFIGS[tree_label]
questions   = ([SAMPLE_QUESTIONS[int(q_idx_str)]] if q_idx_str else SAMPLE_QUESTIONS)

print(f"\n[local] Loading model {model_name} in 4-bit...")
t_load = time.perf_counter()
backend = HFBackend(
    model_name,
    device="cuda",
    load_in_4bit=True,   # fits 1.5B / 1.1B in ~0.75 / 0.55 GB VRAM
)
print(f"[local] Model loaded in {time.perf_counter()-t_load:.1f}s\n")

bf    = tree_cfg["branch_factor"]
depth = tree_cfg["max_depth"]
nodes = (bf**(depth+1) - 1) // (bf - 1)

results = []
for q in questions:
    print(f"\n{'='*60}")
    print(f"  Tree : {tree_label}  ({nodes} nodes, {bf**depth} leaves)")
    print(f"  Q    : {q['question'][:72]}...")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    root = analyse_hf(
        question    = q["question"],
        gold_answer = q["gold_answer"],
        backend     = backend,
        tree_kwargs = {
            "max_depth":     depth,
            "branch_factor": bf,
            "max_tokens":    256,   # shorter steps → faster on CPU/small GPU
            "temperature":   0.8,
            # M-token mode: stop=None (same default as SPO / vLLM script)
        },
        top_k_logprobs = 20,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Done in {elapsed:.1f}s  |  V={root['V']:.3f}")
    print_tree(root)

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.split('/')[-1]}_{tree_label}"

    result = {
        "model":       model_name,
        "tree_config": tree_label,
        "question":    q["question"],
        "gold_answer": q["gold_answer"],
        "elapsed_s":   round(elapsed, 2),
        "root_V":      root["V"],
        "root_P":      root.get("P"),
    }
    results.append(result)

    json_path = save_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  JSON → {json_path}")

    html_path = save_dir / f"{stem}.html"
    visualise_html(
        root,
        path  = str(html_path),
        title = f"{model_name.split('/')[-1]} | {tree_label} | V={root['V']:.3f} | {q['question'][:60]}…",
    )
    print(f"  HTML → {html_path}")

print("\n" + "="*60)
print(f"{'Tree':<8}  {'V':>6}  {'Time(s)':>9}")
print("-"*60)
for r in results:
    print(f"{r['tree_config']:<8}  {r['root_V']:>6.3f}  {r['elapsed_s']:>9.1f}")
print("="*60)
PYEOF
