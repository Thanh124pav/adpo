"""
run_tree_analysis.py
====================
Sample script: run mc_analysis on DeepSeek-R1-Distill-Qwen-1.5B or
Rho-Math-1.1B with three tree configs, then save results as HTML.

Models
------
  DeepSeek-R1-Distill-Qwen-1.5B   deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  Rho-Math-1.1B                   microsoft/rho-math-1.1b-v0.1

Tree structures
---------------
  4-4-4  →  branch_factor=4, max_depth=3  (  85 nodes,  64 leaves)
  6-6-6  →  branch_factor=6, max_depth=3  ( 259 nodes, 216 leaves)
  8-8-8  →  branch_factor=8, max_depth=3  ( 585 nodes, 512 leaves)

Install
-------
  pip install requests numpy vllm

Usage
-----
  # Script starts/stops vLLM automatically
  python run_tree_analysis.py \\
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
      --save-dir ./results/deepseek

  # vLLM already running → skip auto-start
  python run_tree_analysis.py \\
      --server http://localhost:8000/v1 \\
      --no-auto-server \\
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

  # Rho-Math, only 6-6-6, question 0
  python run_tree_analysis.py \\
      --model microsoft/rho-math-1.1b-v0.1 \\
      --tree 6-6-6 --question-idx 0 \\
      --save-dir ./results/rho
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests as _requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mc_analysis import analyse_async, print_tree, visualise_html

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


# ── vLLM server lifecycle ─────────────────────────────────────────────────────

def _server_ready(url: str, timeout: int = 300) -> bool:
    """Poll GET /health until the server responds OK, or timeout expires."""
    health = url.rstrip("/v1").rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _requests.get(health, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def start_vllm(model: str, port: int, max_model_len: int, gpu_memory_utilization: float) -> subprocess.Popen:
    """Launch `vllm serve` and return the Popen handle."""
    cmd = [
        "vllm", "serve", model,
        "--port",                    str(port),
        "--enable-prefix-caching",
        "--max-model-len",           str(max_model_len),
        "--gpu-memory-utilization",  str(gpu_memory_utilization),
        "--trust-remote-code",
    ]
    print(f"\n[server] Starting: {' '.join(cmd)}\n")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,   # new process group so we can kill the whole tree
    )
    return proc


def stop_vllm(proc: subprocess.Popen) -> None:
    """Terminate the vLLM server process group."""
    if proc.poll() is None:
        print("\n[server] Stopping vLLM...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("[server] Stopped.")


# ── analysis ──────────────────────────────────────────────────────────────────

def node_count(b: int, d: int) -> int:
    return (b ** (d + 1) - 1) // (b - 1) if b > 1 else d + 1


async def run_one(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    tree_config: dict,
    tree_label: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    save_dir: Path,
) -> dict:
    bf    = tree_config["branch_factor"]
    depth = tree_config["max_depth"]
    print(f"\n{'='*64}")
    print(f"  Model : {model_name.split('/')[-1]}")
    print(f"  Tree  : {tree_label}  ({node_count(bf, depth)} nodes, {bf**depth} leaves)")
    print(f"  Q     : {question[:72]}...")
    print(f"{'='*64}")

    t0   = time.perf_counter()
    root = await analyse_async(
        question    = question,
        gold_answer = gold_answer,
        server_url  = server_url,
        model_name  = model_name,
        tree_kwargs = {
            "max_depth":     depth,
            "branch_factor": bf,
            "max_tokens":    max_tokens,
            "temperature":   temperature,
        },
        top_k_logprobs = 20,
        max_concurrent = max_concurrent,
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
    stem  = f"{model_name.split('/')[-1]}_{tree_label}"

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


async def run_all(args, server_url: str) -> None:
    configs   = {args.tree: TREE_CONFIGS[args.tree]} if args.tree else TREE_CONFIGS
    questions = (
        [SAMPLE_QUESTIONS[int(args.question_idx)]]
        if args.question_idx is not None
        else SAMPLE_QUESTIONS
    )
    save_dir = Path(args.save_dir)

    print(f"\n  server       : {server_url}")
    print(f"  model        : {args.model}")
    print(f"  tree configs : {list(configs)}")
    print(f"  questions    : {len(questions)}")
    print(f"  save dir     : {save_dir}")

    results = []
    for q in questions:
        for label, cfg in configs.items():
            r = await run_one(
                question      = q["question"],
                gold_answer   = q["gold_answer"],
                server_url    = server_url,
                model_name    = args.model,
                tree_config   = cfg,
                tree_label    = label,
                max_tokens    = args.max_tokens,
                temperature   = args.temperature,
                max_concurrent= args.max_concurrent,
                save_dir      = save_dir,
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
    ap = argparse.ArgumentParser(description="mc_analysis — tree analysis script")
    ap.add_argument("--model",   default=MODELS["deepseek"])
    ap.add_argument("--server",  default="http://localhost:8000/v1",
                    help="vLLM base URL (used when --no-auto-server)")
    ap.add_argument("--port",    type=int,   default=8000)
    ap.add_argument("--no-auto-server", action="store_true",
                    help="Skip auto-start; assume server is already running at --server")
    ap.add_argument("--max-model-len",  type=int,   default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tree",    choices=list(TREE_CONFIGS), default=None)
    ap.add_argument("--question-idx", default=None)
    ap.add_argument("--max-tokens",   type=int,   default=512)
    ap.add_argument("--temperature",  type=float, default=0.8)
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--save-dir", default="./results")
    args = ap.parse_args()

    server_proc = None
    server_url  = args.server

    try:
        if not args.no_auto_server:
            server_proc = start_vllm(
                model                  = args.model,
                port                   = args.port,
                max_model_len          = args.max_model_len,
                gpu_memory_utilization = args.gpu_memory_utilization,
            )
            server_url = f"http://localhost:{args.port}/v1"
            print(f"[server] Waiting for {server_url} to be ready...")
            if not _server_ready(server_url):
                print("[server] ERROR: timed out waiting for vLLM to start.")
                stop_vllm(server_proc)
                sys.exit(1)
            print("[server] Ready.\n")

        asyncio.run(run_all(args, server_url))

    finally:
        if server_proc is not None:
            stop_vllm(server_proc)
