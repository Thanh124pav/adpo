#!/usr/bin/env python3
"""
select_hard_questions.py — Filter dataset to keep only "hard" questions.

A question is "hard" if ALL N rollouts produce a wrong answer (score < threshold).
Useful for building a curriculum dataset focused on the model's weakest points.

Usage:
    python scripts/select_hard_questions.py \
        --model /path/to/model \
        --input data/processed/train/math.parquet \
        --output data/processed/train/math_hard.parquet \
        --n-rollouts 8

    # With 4 GPUs, batched:
    NUM_GPUS=4 python scripts/select_hard_questions.py \
        --model /path/to/model \
        --input data/processed/train/math.parquet \
        --output data/processed/train/math_hard.parquet \
        --n-rollouts 16 --tensor-parallel 4 --batch-size 64

Output parquet preserves all original columns and adds:
    responses      List[str]   — the N generated rollouts
    scores         List[float] — score for each rollout (0.0 / 0.1 / 1.0)
    n_correct      int         — number of correct rollouts (always 0 for hard questions)
    n_rollouts     int         — total rollouts attempted
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

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
# vLLM generation
# ---------------------------------------------------------------------------

def load_model(model_path: str, tensor_parallel: int, gpu_memory_utilization: float):
    from vllm import LLM
    logger.info(f"Loading model {model_path} (tp={tensor_parallel}, gpu_mem={gpu_memory_utilization})")
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def generate_rollouts(
    llm,
    tokenizer,
    prompts: List[List[dict]],
    n_rollouts: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    repetition_penalty: float,
) -> List[List[str]]:
    """Generate n_rollouts responses per prompt. Returns List[List[str]]."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_rollouts,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    formatted = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    outputs = llm.generate(formatted, sampling_params)
    return [[o.text for o in out.outputs] for out in outputs]


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
    """Return prompt as list of message dicts."""
    p = row.get("prompt", [])
    if isinstance(p, list):
        return p
    # numpy array of dicts
    try:
        return list(p)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    # --- Load model ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = load_model(args.model, args.tensor_parallel, args.gpu_memory_utilization)

    # --- Load dataset ---
    df = load_dataset(args.input)
    total = len(df)

    hard_rows = []
    n_processed = 0
    n_hard = 0
    t0 = time.time()

    # Process in batches
    indices = list(range(total))
    batch_size = args.batch_size

    for batch_start in range(0, total, batch_size):
        batch_idx = indices[batch_start: batch_start + batch_size]
        batch = df.iloc[batch_idx]

        prompts = [get_prompt(row) for _, row in batch.iterrows()]
        ground_truths = [get_ground_truth(row) for _, row in batch.iterrows()]
        data_sources = [get_data_source(row) for _, row in batch.iterrows()]

        # Skip rows with no prompt or no ground truth
        valid_mask = [bool(p) and bool(gt) for p, gt in zip(prompts, ground_truths)]
        if not any(valid_mask):
            n_processed += len(batch_idx)
            continue

        # Generate rollouts for valid rows only
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        valid_prompts = [prompts[i] for i in valid_indices]

        rollouts_batch = generate_rollouts(
            llm=llm,
            tokenizer=tokenizer,
            prompts=valid_prompts,
            n_rollouts=args.n_rollouts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        # Score and filter
        for local_i, global_i in enumerate(valid_indices):
            df_row_idx = batch_idx[global_i]
            row = df.iloc[df_row_idx]
            responses = rollouts_batch[local_i]
            gt = ground_truths[global_i]
            src = data_sources[global_i]

            scores = [score_response(src, r, gt) for r in responses]
            n_correct = sum(1 for s in scores if s >= args.correct_threshold)

            if n_correct == 0:  # all wrong → hard question
                record = row.to_dict()
                record["responses"] = responses
                record["scores"] = scores
                record["n_correct"] = n_correct
                record["n_rollouts"] = args.n_rollouts
                hard_rows.append(record)
                n_hard += 1

        n_processed += len(batch_idx)
        elapsed = time.time() - t0
        rate = n_processed / elapsed
        eta = (total - n_processed) / rate if rate > 0 else 0

        logger.info(
            f"[{n_processed}/{total}] hard={n_hard} "
            f"({100*n_hard/max(n_processed,1):.1f}%)  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    # --- Save ---
    if not hard_rows:
        logger.warning("No hard questions found. Output file not written.")
        return

    out_df = pd.DataFrame(hard_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    logger.info(
        f"\nDone. {n_hard}/{total} hard questions "
        f"({100*n_hard/total:.1f}%) saved to {args.output}"
    )
    logger.info(f"Output columns: {list(out_df.columns)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select hard questions where all N rollouts are wrong.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--model", required=True,
                        help="Path to model (HF or local)")
    parser.add_argument("--input", required=True,
                        help="Input parquet file (verl format)")
    parser.add_argument("--output", required=True,
                        help="Output parquet file for hard questions")

    # Rollout
    parser.add_argument("--n-rollouts", type=int, default=8,
                        help="Number of rollouts per question")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per rollout")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Repetition penalty")

    # Scoring
    parser.add_argument("--correct-threshold", type=float, default=1.0,
                        help="Score >= this is considered correct (e.g. 1.0 for exact match)")

    # Hardware
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of prompts to send to vLLM per batch")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
