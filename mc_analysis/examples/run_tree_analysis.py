"""
run_tree_analysis.py
====================
Sample script: run mc_analysis on two small models with three tree configs.

Models
------
  DeepSeek-R1-Distill-Qwen-1.5B   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  Rho-Math-1.1B                   microsoft/rho-math-1.1b-v0.1

Tree structures
---------------
  4-4-4  →  branch_factor=4, max_depth=3  (  85 nodes,  64 leaves)
  6-6-6  →  branch_factor=6, max_depth=3  ( 259 nodes, 216 leaves)
  8-8-8  →  branch_factor=8, max_depth=3  ( 585 nodes, 512 leaves)

Prerequisites
-------------
  1. Start a vLLM server with prefix-caching enabled, e.g.:
       python -m vllm.entrypoints.openai.api_server \\
           --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
           --port 8000 \\
           --enable-prefix-caching \\
           --max-model-len 8192

  2. Install: pip install requests numpy matplotlib

Usage
-----
  # vLLM server already running on localhost:8000
  python run_tree_analysis.py

  # Custom server / model
  python run_tree_analysis.py --server http://gpu-node:8001/v1 \\
                               --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
                               --tree 6-6-6

  # Save plots instead of showing them
  python run_tree_analysis.py --save-dir ./results
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# ── allow running from repo root without installing the package ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mc_analysis import analyse_async, print_tree, visualise

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
        "question": (
            "Find the largest prime factor of $9879$."
        ),
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

# ── tree configurations ───────────────────────────────────────────────────────
TREE_CONFIGS = {
    "4-4-4": {"branch_factor": 4, "max_depth": 3},
    "6-6-6": {"branch_factor": 6, "max_depth": 3},
    "8-8-8": {"branch_factor": 8, "max_depth": 3},
}

# ── models ────────────────────────────────────────────────────────────────────
MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "rho":      "microsoft/rho-math-1.1b-v0.1",
}


def node_count(branch_factor: int, max_depth: int) -> int:
    """Total nodes in a complete B-ary tree of given depth."""
    if branch_factor == 1:
        return max_depth + 1
    return (branch_factor ** (max_depth + 1) - 1) // (branch_factor - 1)


async def run_one(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    tree_config: dict,
    tree_label: str,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_concurrent: int = 16,
    save_dir: Path = None,
    show_plot: bool = True,
) -> dict:
    bf   = tree_config["branch_factor"]
    depth = tree_config["max_depth"]
    n_nodes = node_count(bf, depth)
    n_leaves = bf ** depth

    print(f"\n{'='*60}")
    print(f"  Model : {model_name.split('/')[-1]}")
    print(f"  Tree  : {tree_label}  ({n_nodes} nodes, {n_leaves} leaves)")
    print(f"  Q     : {question[:70]}...")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    root = await analyse_async(
        question      = question,
        gold_answer   = gold_answer,
        server_url    = server_url,
        model_name    = model_name,
        tree_kwargs   = {
            "max_depth":     depth,
            "branch_factor": bf,
            "max_tokens":    max_tokens,
            "temperature":   temperature,
            "top_p":         top_p,
        },
        top_k_logprobs  = 20,
        max_concurrent  = max_concurrent,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Root V = {root['V']:.3f}   Root P = {root.get('P', float('nan')):.2f}")
    print()

    print_tree(root)

    # ── save JSON ─────────────────────────────────────────────────────────────
    result = {
        "model":       model_name,
        "tree":        tree_label,
        "question":    question,
        "gold_answer": gold_answer,
        "elapsed_s":   round(elapsed, 2),
        "root_V":      root["V"],
        "root_P":      root.get("P"),
        "tree":        root,   # full annotated tree (JSON-serialisable)
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_short = model_name.split("/")[-1]
        stem = f"{model_short}_{tree_label}"

        json_path = save_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved JSON → {json_path}")

        fig_path = save_dir / f"{stem}.png"
        visualise(
            root,
            title=f"{model_short} | {tree_label} | V={root['V']:.2f}",
            show=show_plot,
            save_path=str(fig_path),
        )
        print(f"  Saved plot → {fig_path}")
    elif show_plot:
        visualise(
            root,
            title=f"{model_name.split('/')[-1]} | {tree_label} | V={root['V']:.2f}",
            show=True,
        )

    return result


async def main(args):
    server_url  = args.server
    model_name  = args.model
    save_dir    = Path(args.save_dir) if args.save_dir else None
    show_plot   = not args.no_plot

    # Which tree configs to run
    if args.tree:
        if args.tree not in TREE_CONFIGS:
            print(f"Unknown tree config '{args.tree}'. Choose from: {list(TREE_CONFIGS)}")
            sys.exit(1)
        configs = {args.tree: TREE_CONFIGS[args.tree]}
    else:
        configs = TREE_CONFIGS

    # Which questions to run
    questions = SAMPLE_QUESTIONS if not args.question_idx else \
                [SAMPLE_QUESTIONS[int(args.question_idx)]]

    print(f"\nvLLM server  : {server_url}")
    print(f"Model        : {model_name}")
    print(f"Tree configs : {list(configs)}")
    print(f"Questions    : {len(questions)}")
    print(f"\n⚠  Ensure the server was started with --enable-prefix-caching")

    all_results = []
    for q in questions:
        for label, cfg in configs.items():
            r = await run_one(
                question      = q["question"],
                gold_answer   = q["gold_answer"],
                server_url    = server_url,
                model_name    = model_name,
                tree_config   = cfg,
                tree_label    = label,
                max_tokens    = args.max_tokens,
                temperature   = args.temperature,
                max_concurrent= args.max_concurrent,
                save_dir      = save_dir,
                show_plot     = show_plot,
            )
            all_results.append(r)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Tree':<8} {'V':>6} {'P':>8} {'Time(s)':>9}")
    print("-"*60)
    for r in all_results:
        p_str = f"{r['root_P']:.1f}" if r["root_P"] is not None else "N/A"
        print(f"{r['tree']:<8} {r['root_V']:>6.3f} {p_str:>8} {r['elapsed_s']:>9.1f}")
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="mc_analysis tree analysis script")

    p.add_argument(
        "--server", default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)",
    )
    p.add_argument(
        "--model", default=MODELS["deepseek"],
        help=f"Model name on the vLLM server (default: {MODELS['deepseek']})",
    )
    p.add_argument(
        "--tree", choices=list(TREE_CONFIGS), default=None,
        help="Run a single tree config (default: run all three)",
    )
    p.add_argument(
        "--question-idx", default=None,
        help="Index into sample questions (0/1/2). Default: run all.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens per generation step (default: 512)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    p.add_argument(
        "--max-concurrent", type=int, default=16,
        help="Max concurrent vLLM requests (default: 16)",
    )
    p.add_argument(
        "--save-dir", default=None,
        help="Directory to save JSON results and PNG plots",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Disable interactive matplotlib windows",
    )

    args = p.parse_args()
    asyncio.run(main(args))
