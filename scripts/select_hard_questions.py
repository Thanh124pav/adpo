#!/usr/bin/env python3
"""
select_hard_questions.py — Filter dataset to keep only "hard" questions.

A question is "hard" if ALL N rollouts produce a wrong answer (score < threshold).
Useful for building a curriculum dataset focused on the model's weakest points.

Two inference backends (--backend):
    local     — load model via vLLM on local GPUs
    endpoint  — send requests to an OpenAI-compatible API (vLLM server, etc.)

Output:
    --output  <file>.parquet   — hard questions, original columns only (no extra)
    --meta    <file>.jsonl     — one JSON line per hard question with rollout metadata:
                                 {dataset_index, data_source, ground_truth,
                                  responses, scores, n_correct, n_rollouts}

Usage:
    # Local vLLM
    python scripts/select_hard_questions.py \
        --backend local \
        --model /raid/models/R1-Distill-1.5B \
        --input  data/processed/train/math.parquet \
        --output data/processed/train/math_hard.parquet \
        --meta   data/processed/train/math_hard_meta.jsonl \
        --n-rollouts 8

    # Remote endpoint (vLLM OpenAI-compatible server)
    python scripts/select_hard_questions.py \
        --backend endpoint \
        --endpoint http://localhost:8000 \
        --endpoint-model Qwen3-4B \
        --input  data/processed/train/math.parquet \
        --output data/processed/train/math_hard.parquet \
        --meta   data/processed/train/math_hard_meta.jsonl \
        --n-rollouts 16 --batch-size 64
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_response(data_source: str, response: str, ground_truth: str) -> float:
    try:
        from adpo.reward_functions import compute_score
        return float(compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
        ))
    except Exception as e:
        logger.warning(f"compute_score failed ({e}), returning 0.0")
        return 0.0


# ---------------------------------------------------------------------------
# Backend: local vLLM
# ---------------------------------------------------------------------------

class LocalBackend:
    def __init__(self, args):
        from vllm import LLM
        from transformers import AutoTokenizer
        logger.info(
            f"Loading model {args.model} "
            f"(tp={args.tensor_parallel}, gpu_mem={args.gpu_memory_utilization})"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        self.args = args

    def generate(self, prompts: List[List[dict]]) -> List[List[str]]:
        """Returns List[List[str]]: n_rollouts responses per prompt."""
        from vllm import SamplingParams
        a = self.args
        sampling_params = SamplingParams(
            temperature=a.temperature,
            max_tokens=a.max_tokens,
            n=a.n_rollouts,
            top_p=a.top_p,
            repetition_penalty=a.repetition_penalty,
        )
        formatted = [
            self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        outputs = self.llm.generate(formatted, sampling_params)
        return [[o.text for o in out.outputs] for out in outputs]


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

class EndpointBackend:
    """Calls an OpenAI-compatible /v1/chat/completions endpoint.

    Sends one request per prompt (n=N_rollouts). Concurrent requests via
    ThreadPoolExecutor for throughput.
    """

    def __init__(self, args):
        self.endpoint = args.endpoint.rstrip("/")
        self.model_name = args.endpoint_model
        self.args = args
        logger.info(f"Using endpoint {self.endpoint}  model={self.model_name}")

    def _call_one(self, idx: int, messages: List[dict], total: int) -> tuple:
        """Single /v1/chat/completions call. Returns (idx, List[str])."""
        import urllib.request, urllib.error
        a = self.args
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "n": a.n_rollouts,
            "temperature": a.temperature,
            "max_tokens": a.max_tokens,
            "top_p": a.top_p,
        }).encode()

        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
                elapsed = time.time() - t0
                logger.debug(f"  req {idx+1}/{total} done in {elapsed:.1f}s")
                return idx, [choice["message"]["content"] for choice in data["choices"]]
            except urllib.error.URLError as e:
                wait = 2 ** attempt
                logger.warning(f"  req {idx+1}/{total} failed ({e}), retry in {wait}s...")
                time.sleep(wait)
        logger.error(f"  req {idx+1}/{total} exhausted all retries, returning empty.")
        return idx, [""] * a.n_rollouts

    def generate(self, prompts: List[List[dict]]) -> List[List[str]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        total = len(prompts)
        results = [None] * total
        n_done = 0
        t0 = time.time()
        logger.info(f"  Sending {total} requests (workers={self.args.endpoint_workers}, n_rollouts={self.args.n_rollouts})...")
        with ThreadPoolExecutor(max_workers=self.args.endpoint_workers) as pool:
            futures = {pool.submit(self._call_one, i, p, total): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                idx, responses = fut.result()
                results[idx] = responses
                n_done += 1
                if n_done % max(1, total // 10) == 0 or n_done == total:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (total - n_done) / rate
                    logger.info(f"  [{n_done}/{total}] {elapsed:.0f}s elapsed  ETA={eta:.0f}s")
        return results


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} samples from {path}  columns={list(df.columns)}")
    return df


def get_ground_truth(row) -> str:
    rm = row.get("reward_model", {})
    if isinstance(rm, dict):
        return rm.get("ground_truth", "")
    return ""


def get_data_source(row) -> str:
    return str(row.get("data_source", ""))


def get_prompt(row) -> List[dict]:
    p = row.get("prompt", [])
    if isinstance(p, list):
        return p
    try:
        return list(p)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    # --- Backend ---
    if args.backend == "local":
        backend = LocalBackend(args)
    else:
        if not args.endpoint:
            raise ValueError("--endpoint is required for --backend endpoint")
        if not args.endpoint_model:
            raise ValueError("--endpoint-model is required for --backend endpoint")
        backend = EndpointBackend(args)

    # --- Dataset ---
    df = load_dataset(args.input)
    total = len(df)

    hard_df_indices: List[int] = []   # row positions in df → for parquet output
    meta_records: List[dict] = []     # rollout metadata → for JSONL output

    n_processed = 0
    n_hard = 0
    t0 = time.time()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_fh = meta_path.open("a", encoding="utf-8")

    try:
        for batch_start in range(0, total, args.batch_size):
            batch_idx = list(range(batch_start, min(batch_start + args.batch_size, total)))
            batch = df.iloc[batch_idx]

            prompts = [get_prompt(row) for _, row in batch.iterrows()]
            ground_truths = [get_ground_truth(row) for _, row in batch.iterrows()]
            data_sources = [get_data_source(row) for _, row in batch.iterrows()]

            # Only process rows with both prompt and ground truth
            valid_indices = [
                i for i, (p, gt) in enumerate(zip(prompts, ground_truths))
                if bool(p) and bool(gt)
            ]
            if not valid_indices:
                n_processed += len(batch_idx)
                continue

            rollouts_batch = backend.generate([prompts[i] for i in valid_indices])

            for local_i, batch_local_i in enumerate(valid_indices):
                df_row_idx = batch_idx[batch_local_i]
                responses = rollouts_batch[local_i]
                gt = ground_truths[batch_local_i]
                src = data_sources[batch_local_i]

                scores = [score_response(src, r, gt) for r in responses]
                n_correct = sum(1 for s in scores if s >= args.correct_threshold)

                if n_correct == 0:  # hard question
                    hard_df_indices.append(df_row_idx)

                    meta = {
                        "dataset_index": df_row_idx,
                        "data_source": src,
                        "ground_truth": gt,
                        "responses": responses,
                        "scores": scores,
                        "n_correct": n_correct,
                        "n_rollouts": args.n_rollouts,
                    }
                    meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    meta_fh.flush()
                    n_hard += 1

            n_processed += len(batch_idx)
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-6)
            eta = (total - n_processed) / rate
            logger.info(
                f"[{n_processed}/{total}] hard={n_hard} "
                f"({100 * n_hard / max(n_processed, 1):.1f}%)  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )
    finally:
        meta_fh.close()

    # --- Save parquet (original columns only) ---
    if not hard_df_indices:
        logger.warning("No hard questions found. Parquet not written.")
        return

    hard_df = df.iloc[hard_df_indices].reset_index(drop=True)
    hard_df.to_parquet(args.output, index=False)

    logger.info(
        f"\nDone. {n_hard}/{total} hard questions ({100 * n_hard / total:.1f}%)"
    )
    logger.info(f"  Parquet → {args.output}  ({list(hard_df.columns)})")
    logger.info(f"  Meta    → {args.meta}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select hard questions where all N rollouts are wrong.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--input",  required=True, help="Input parquet file (verl format)")
    parser.add_argument("--output", required=True, help="Output parquet (original columns only)")
    parser.add_argument("--meta",   required=True, help="Output JSONL for rollout metadata")

    # Backend
    parser.add_argument("--backend", choices=["local", "endpoint"], default="local",
                        help="local = vLLM on GPUs; endpoint = OpenAI-compatible API")

    # Local backend
    parser.add_argument("--model", default="",
                        help="[local] Model path (HF or local dir)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="[local] Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="[local] vLLM GPU memory utilization")

    # Endpoint backend
    parser.add_argument("--endpoint", default="",
                        help="[endpoint] Base URL, e.g. http://localhost:8000")
    parser.add_argument("--endpoint-model", default="",
                        help="[endpoint] Model name registered at the endpoint")
    parser.add_argument("--endpoint-workers", type=int, default=16,
                        help="[endpoint] Concurrent requests to the endpoint")

    # Sampling
    parser.add_argument("--n-rollouts",          type=int,   default=8)
    parser.add_argument("--temperature",          type=float, default=0.7)
    parser.add_argument("--max-tokens",           type=int,   default=4096)
    parser.add_argument("--top-p",                type=float, default=0.95)
    parser.add_argument("--repetition-penalty",   type=float, default=1.05)

    # Scoring
    parser.add_argument("--correct-threshold", type=float, default=1.0,
                        help="score >= this counts as correct")

    # Batching
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Prompts per batch")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
