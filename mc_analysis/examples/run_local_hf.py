"""
run_local_hf.py
===============
Run mc_analysis with the HuggingFace backend (no vLLM server needed).
Designed for machines with limited VRAM (≥4 GB) using 4-bit quantization.

Models
------
  DeepSeek-R1-Distill-Qwen-1.5B   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  Rho-Math-1.1B                   microsoft/rho-math-1.1b-v0.1

Tree structures
---------------
  4-4-4  →  branch_factor=4, max_depth=3  ( 85 nodes,  64 leaves)
  6-6-6  →  branch_factor=6, max_depth=3  (259 nodes, 216 leaves)
  8-8-8  →  branch_factor=8, max_depth=3  (585 nodes, 512 leaves)

Install
-------
  pip install torch transformers bitsandbytes accelerate numpy pandas pyarrow

Usage
-----
  # Built-in questions, tree 4-4-4, int4 quantization
  python run_local_hf.py \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --tree 4-4-4 --load-in-4bit

  # Load 8 random examples from a parquet dataset
  python run_local_hf.py \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --parquet /path/to/gsm8k.parquet --num-examples 8 --seed 7 \
      --tree 4-4-4 --load-in-4bit

  # fp16, all trees, single question
  python run_local_hf.py \
      --model microsoft/rho-math-1.1b-v0.1 \
      --tree 6-6-6 --question-idx 1
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mc_analysis import HFBackend, analyse_hf, print_tree, visualise_html

# ── sample questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning "
            "and bakes muffins for her friends every day with 4. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. "
            "How much in dollars does she make every day at the farmers' market?"
        ),
        "gold_answer": "18",
        "source": "GSM8K",
    },
    {
        "question": "Find the largest prime factor of $9879$.",
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

MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "rho":      "microsoft/rho-math-1.1b-v0.1",
}


# ── parquet loader ────────────────────────────────────────────────────────────

def load_parquet_examples(path: str, n: int, seed: int = 42) -> List[dict]:
    """Load *n* random examples from a verl-format parquet file."""
    try:
        import pandas as pd
    except ImportError:
        print("[error] pandas not installed. Run: pip install pandas pyarrow")
        sys.exit(1)

    df = pd.read_parquet(path)
    if len(df) == 0:
        print(f"[warn] Parquet file is empty: {path}")
        return []

    rng = random.Random(seed)
    indices = rng.sample(range(len(df)), min(n, len(df)))
    rows = df.iloc[indices]

    examples = []
    for _, row in rows.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        question = ""
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg["content"]
                break

        rm = row["reward_model"]
        gold_answer = str(rm.get("ground_truth", "")) if isinstance(rm, dict) else str(rm)

        examples.append({
            "question":    question,
            "gold_answer": gold_answer,
            "source":      str(row.get("data_source", "parquet")),
        })

    print(f"[parquet] Loaded {len(examples)} examples from {path}  (seed={seed})")
    return examples


# ── helpers ───────────────────────────────────────────────────────────────────

def node_count(b: int, d: int) -> int:
    return (b ** (d + 1) - 1) // (b - 1) if b > 1 else d + 1


def run_one(
    question:    str,
    gold_answer: str,
    backend:     HFBackend,
    model_name:  str,
    tree_config: dict,
    tree_label:  str,
    max_tokens:  int,
    temperature: float,
    save_dir:    Path,
) -> dict:
    bf    = tree_config["branch_factor"]
    depth = tree_config["max_depth"]
    print(f"\n{'='*64}")
    print(f"  Model : {model_name.split('/')[-1]}")
    print(f"  Tree  : {tree_label}  ({node_count(bf, depth)} nodes, {bf**depth} leaves)")
    print(f"  Q     : {question[:72]}...")
    print(f"{'='*64}")

    t0   = time.perf_counter()
    root = analyse_hf(
        question    = question,
        gold_answer = gold_answer,
        backend     = backend,
        tree_kwargs = {
            "max_depth":     depth,
            "branch_factor": bf,
            "max_tokens":    max_tokens,
            "temperature":   temperature,
            # M-token splitting mode (stop=None, same default as SPO / vLLM script)
        },
        top_k_logprobs = 20,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Done in {elapsed:.1f}s  |  V={root['V']:.3f}  P={root.get('P', float('nan')):.2f}")
    print_tree(root)

    result = {
        "model":       model_name,
        "tree_config": tree_label,
        "question":    question,
        "gold_answer": gold_answer,
        "elapsed_s":   round(elapsed, 2),
        "root_V":      root["V"],
        "root_P":      root.get("P"),
        "root":        root,
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.split('/')[-1]}_{tree_label}"

    json_path = save_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  JSON  → {json_path}")

    html_path = save_dir / f"{stem}.html"
    visualise_html(
        root,
        path  = str(html_path),
        title = f"{model_name.split('/')[-1]} | {tree_label} | V={root['V']:.3f} | {question[:60]}…",
    )
    print(f"  HTML  → {html_path}")

    return result


def run_all(args) -> None:
    configs = {args.tree: TREE_CONFIGS[args.tree]} if args.tree else TREE_CONFIGS

    if args.parquet:
        questions = load_parquet_examples(args.parquet, args.num_examples, args.seed)
    elif args.question_idx is not None:
        questions = [SAMPLE_QUESTIONS[int(args.question_idx)]]
    else:
        questions = SAMPLE_QUESTIONS

    save_dir = Path(args.save_dir)

    quant = "int4" if args.load_in_4bit else ("int8" if args.load_in_8bit else "fp16")
    print(f"\n  backend      : HuggingFace ({quant})")
    print(f"  model        : {args.model}")
    print(f"  tree configs : {list(configs)}")
    print(f"  questions    : {len(questions)}")
    print(f"  save dir     : {save_dir}")

    print(f"\n[hf] Loading {args.model} ({quant})...")
    t_load = time.perf_counter()
    backend = HFBackend(
        args.model,
        device       = args.device,
        load_in_4bit = args.load_in_4bit,
        load_in_8bit = args.load_in_8bit,
    )
    print(f"[hf] Model loaded in {time.perf_counter()-t_load:.1f}s\n")

    results = []
    for q in questions:
        for label, cfg in configs.items():
            r = run_one(
                question    = q["question"],
                gold_answer = q["gold_answer"],
                backend     = backend,
                model_name  = args.model,
                tree_config = cfg,
                tree_label  = label,
                max_tokens  = args.max_tokens,
                temperature = args.temperature,
                save_dir    = save_dir,
            )
            results.append(r)

    print("\n" + "="*64)
    print(f"{'Tree':<8}  {'V':>6}  {'P':>8}  {'Time(s)':>9}")
    print("-"*64)
    for r in results:
        p_str = f"{r['root_P']:.1f}" if r["root_P"] is not None else "N/A"
        print(f"{r['tree_config']:<8}  {r['root_V']:>6.3f}  {p_str:>8}  {r['elapsed_s']:>9.1f}")
    print("="*64)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="mc_analysis — HF backend (local GPU)")
    ap.add_argument("--model",        default=MODELS["deepseek"])
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load model in 4-bit (NF4) — fits ~1.5B in 1 GB VRAM. "
                         "Requires bitsandbytes.")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="Load model in 8-bit — intermediate between fp16 and int4.")
    ap.add_argument("--tree",         choices=list(TREE_CONFIGS), default=None)
    ap.add_argument("--question-idx", default=None)
    ap.add_argument("--max-tokens",   type=int,   default=256,
                    help="Tokens per step (default 256; lower = faster on small GPU).")
    ap.add_argument("--temperature",  type=float, default=0.8)
    ap.add_argument("--save-dir",     default="./results")
    ap.add_argument("--parquet",      default=None,
                    help="Path to a verl-format parquet file. "
                         "Randomly sample --num-examples rows from it.")
    ap.add_argument("--num-examples", type=int,   default=5,
                    help="Number of random examples to sample from --parquet (default: 5).")
    ap.add_argument("--seed",         type=int,   default=42,
                    help="Random seed for parquet sampling (default: 42).")
    args = ap.parse_args()

    run_all(args)
