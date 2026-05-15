"""
run_api.py
==========
Run mc_analysis against an external OpenAI-compatible API endpoint
(OpenAI, Together AI, DeepSeek, Fireworks, etc.) — no local GPU required.

Key differences from run_tree_analysis.py
------------------------------------------
* No server start/stop — requests go directly to the external endpoint.
* No nvidia-smi zombie cleanup (not applicable).
* Defaults to --compute-p-inline (echo-mode is not supported by most APIs).
* API key passed via --api-key or the API_KEY environment variable.

Supported providers (set --api-url accordingly)
-------------------------------------------------
  OpenAI        https://api.openai.com/v1
  Together AI   https://api.together.xyz/v1
  Fireworks AI  https://api.fireworks.ai/inference/v1
  DeepSeek      https://api.deepseek.com/v1
  Local vLLM    http://localhost:8000/v1   (no API key needed)

Usage
-----
  # Together AI — DeepSeek-R1 (free tier)
  python run_api.py \\
      --api-url  https://api.together.xyz/v1 \\
      --api-key  $TOGETHER_API_KEY \\
      --model    deepseek-ai/DeepSeek-R1 \\
      --tree     4-4-4

  # OpenAI — GPT-4o-mini, parquet file
  python run_api.py \\
      --api-url  https://api.openai.com/v1 \\
      --api-key  $OPENAI_API_KEY \\
      --model    gpt-4o-mini \\
      --parquet  ./data/processed/train/gsm8k.parquet \\
      --num-examples 10

  # Local vLLM already running (no API key)
  python run_api.py \\
      --api-url  http://localhost:8000/v1 \\
      --model    /workspace/storage-shared/models/DeepSeek-R1-Distill-Qwen-1.5B
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mc_analysis import analyse_async, print_tree, visualise_html
from _utils import append_summary, load_parquet_examples, make_stem

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


# ── analysis ──────────────────────────────────────────────────────────────────

def node_count(b: int, d: int) -> int:
    return (b ** (d + 1) - 1) // (b - 1) if b > 1 else d + 1


async def run_one(
    question: str,
    gold_answer: str,
    source: str,
    server_url: str,
    model_name: str,
    api_key: str,
    tree_config: dict,
    tree_label: str,
    max_tokens: int,
    temperature: float,
    stop,
    max_concurrent: int,
    score_concurrent: int,
    compute_p_inline: bool,
    p_max_tokens: int,
    save_dir: Path,
) -> dict:
    bf    = tree_config["branch_factor"]
    depth = tree_config["max_depth"]
    print(f"\n{'='*64}")
    print(f"  Model : {model_name.split('/')[-1]}")
    print(f"  API   : {server_url}")
    print(f"  Tree  : {tree_label}  ({node_count(bf, depth)} nodes, {bf**depth} leaves)")
    print(f"  Q     : {question[:72]}...")
    print(f"{'='*64}")

    t0   = time.perf_counter()
    root = await analyse_async(
        question    = question,
        gold_answer = gold_answer,
        server_url  = server_url,
        model_name  = model_name,
        api_key     = api_key,
        tree_kwargs = {
            "max_depth":     depth,
            "branch_factor": bf,
            "max_tokens":    max_tokens,
            "temperature":   temperature,
            "stop":          stop or None,
            "p_max_tokens":  p_max_tokens,
        },
        top_k_logprobs   = 20,
        max_concurrent   = max_concurrent,
        score_concurrent = score_concurrent,
        # echo=True is not supported by most external APIs;
        # inline P fires a free-form completion per node instead.
        compute_p        = False,
        compute_p_inline = compute_p_inline,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Done in {elapsed:.1f}s  |  V={root['V']:.3f}  P={root.get('P', float('nan')):.2f}")
    print_tree(root)

    result = {
        "model":       model_name,
        "api_url":     server_url,
        "tree_config": tree_label,
        "source":      source,
        "question":    question,
        "gold_answer": gold_answer,
        "elapsed_s":   round(elapsed, 2),
        "root_V":      root["V"],
        "root_P":      root.get("P"),
        "root":        root,
    }

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = make_stem(model_name, tree_label, question)

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

    append_summary(save_dir, result, stem)
    return result


def run_all(args, server_url: str, api_key: str) -> None:
    configs = {args.tree: TREE_CONFIGS[args.tree]} if args.tree else TREE_CONFIGS

    if args.parquet:
        questions = load_parquet_examples(args.parquet, args.num_examples, args.seed)
    elif args.question_idx is not None:
        questions = [SAMPLE_QUESTIONS[int(args.question_idx)]]
    else:
        questions = SAMPLE_QUESTIONS

    save_dir = Path(args.save_dir)
    items    = [(q, label, cfg) for q in questions for label, cfg in configs.items()]

    print(f"\n  api url      : {server_url}")
    print(f"  model        : {args.model}")
    print(f"  tree configs : {list(configs)}")
    print(f"  questions    : {len(questions)}")
    print(f"  save dir     : {save_dir}")
    print(f"  inline P     : {args.compute_p_inline}")

    results = []
    for q, label, cfg in items:
        r = asyncio.run(run_one(
            question         = q["question"],
            gold_answer      = q["gold_answer"],
            source           = q.get("source", ""),
            server_url       = server_url,
            model_name       = args.model,
            api_key          = api_key,
            tree_config      = cfg,
            tree_label       = label,
            max_tokens       = args.max_tokens,
            temperature      = args.temperature,
            stop             = args.stop,
            max_concurrent   = args.max_concurrent,
            score_concurrent = args.score_concurrent,
            compute_p_inline = args.compute_p_inline,
            p_max_tokens     = args.p_max_tokens,
            save_dir         = save_dir,
        ))
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
    ap = argparse.ArgumentParser(description="mc_analysis — external API endpoint")
    ap.add_argument("--api-url", required=True,
                    help="Base URL of an OpenAI-compatible API "
                         "(e.g. https://api.together.xyz/v1)")
    ap.add_argument("--api-key", default=None,
                    help="API key. If omitted, reads from the API_KEY environment variable.")
    ap.add_argument("--model", required=True,
                    help="Full model ID on the API (e.g. deepseek-ai/DeepSeek-R1)")
    ap.add_argument("--tree", choices=list(TREE_CONFIGS), default=None)
    ap.add_argument("--question-idx", default=None)
    ap.add_argument("--max-tokens",   type=int,   default=512)
    ap.add_argument("--temperature",  type=float, default=0.8)
    ap.add_argument("--stop", nargs="*", default=[],
                    help="Stop sequences (default: empty = M-token mode).")
    ap.add_argument("--max-concurrent",   type=int, default=4,
                    help="Concurrency for generation requests. "
                         "Keep low for rate-limited APIs (default 4).")
    ap.add_argument("--score-concurrent", type=int, default=2,
                    help="Concurrency for JSD annotation requests (default 2).")
    ap.add_argument("--compute-p-inline", action="store_true", default=True,
                    help="Compute P inline via answer branch (default: on). "
                         "External APIs do not support echo mode.")
    ap.add_argument("--no-compute-p-inline", dest="compute_p_inline", action="store_false",
                    help="Disable inline P (skips P entirely for this run).")
    ap.add_argument("--p-max-tokens", type=int, default=1024,
                    help="Max tokens for the inline P answer branch (default: 1024).")
    ap.add_argument("--save-dir", default="./results/api")
    ap.add_argument("--parquet", default=None,
                    help="Path to a parquet file; randomly sample --num-examples rows.")
    ap.add_argument("--num-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("API_KEY", "")
    if not api_key and not args.api_url.startswith("http://localhost"):
        print("[warn] No API key provided (--api-key or API_KEY env var). "
              "Requests may be rejected by the endpoint.")

    run_all(args, server_url=args.api_url, api_key=api_key)
