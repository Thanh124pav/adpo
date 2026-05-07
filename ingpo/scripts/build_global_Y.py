"""Pre-compute a single global Y for Abl 5 (Y-per-dataset).

Reads a JSONL with the train problems (one per line, fields: problem,
answer), prompts the served vLLM with PLAN.md's Y-template, and writes a
single JSON file with `m` step-by-step solutions.

Usage:
    python scripts/build_global_Y.py \
        --train_jsonl data/MATH/train.jsonl \
        --api_base http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen2.5-1.5B \
        --m 100 \
        --out data/MATH/global_Y.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import List

import openai

from ingpo_ext.core.answer_set import DEFAULT_Y_PROMPT_TEMPLATE


async def sample_completions(client, model: str, prompt: str, n: int, max_tokens: int) -> List[str]:
    resp = await client.completions.create(
        model=model,
        prompt=prompt,
        n=n,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return [c.text for c in resp.choices]


async def main_async(args):
    train = [json.loads(l) for l in Path(args.train_jsonl).read_text().splitlines() if l.strip()]
    random.seed(args.seed)
    random.shuffle(train)
    seed = train[0]
    prompt = DEFAULT_Y_PROMPT_TEMPLATE.format(problem=seed["problem"], gold=seed.get("answer", ""))

    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=args.api_base)
    completions = await sample_completions(client, args.model, prompt, args.m, args.max_tokens)
    Path(args.out).write_text(json.dumps({"prompt": prompt, "y": completions}, indent=2))
    print(f"Wrote {len(completions)} answers to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
