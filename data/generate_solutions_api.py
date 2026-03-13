"""
Generate reference solutions using an OpenAI-compatible API endpoint.

Two strategies:
1. "hint": Given the correct answer, generate step-by-step reasoning
2. "free": Generate solutions freely (no answer given), verify correctness,
   keep only correct ones

Uses async HTTP for high throughput.

Usage:
    # Strategy 1: Hint-based (answer provided)
    python data/generate_solutions_api.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions.jsonl \
        --strategy hint \
        --num_solutions 4

    # Strategy 2: Free generation + verification
    python data/generate_solutions_api.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions.jsonl \
        --strategy free \
        --num_solutions 8

    # Custom endpoint
    python data/generate_solutions_api.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions.jsonl \
        --endpoint http://10.254.138.189:8104 \
        --model Qwen/Qwen2.5-72B-Instruct \
        --strategy hint
"""

import argparse
import asyncio
import json
import os
import re
import hashlib
import logging
import time
from typing import List, Dict, Optional, Tuple

import pandas as pd
import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HINT_SYSTEM_PROMPT = (
    "You are a math problem solver. You are given a math problem and its correct answer. "
    "Write a detailed step-by-step solution that arrives at the given answer.\n\n"
    "IMPORTANT formatting rules:\n"
    "- Use '.' ONLY at the very end of your complete solution to mark the finish\n"
    "- Do NOT use '.' anywhere else in your reasoning (use commas, semicolons, "
    "colons, or line breaks instead)\n"
    "- Put your final answer within \\boxed{}"
)

FREE_SYSTEM_PROMPT = (
    "Please reason step by step to solve the following math problem.\n\n"
    "IMPORTANT formatting rules:\n"
    "- Use '.' ONLY at the very end of your complete solution to mark the finish\n"
    "- Do NOT use '.' anywhere else in your reasoning (use commas, semicolons, "
    "colons, or line breaks instead)\n"
    "- Put your final answer within \\boxed{}"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_question_id(question: str) -> str:
    """Deterministic ID for a question."""
    return hashlib.sha256(question.strip().encode()).hexdigest()[:16]


def extract_question_from_prompt(prompt_json: str) -> str:
    """Extract question text from verl prompt format."""
    try:
        messages = json.loads(prompt_json)
        for msg in messages:
            if msg["role"] == "user":
                return msg["content"]
    except (json.JSONDecodeError, KeyError):
        pass
    return prompt_json


def extract_ground_truth(reward_model_json: str) -> str:
    """Extract ground truth from verl reward_model format."""
    try:
        data = json.loads(reward_model_json)
        return data.get("ground_truth", "")
    except (json.JSONDecodeError, KeyError):
        return reward_model_json


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace("\\frac", "frac")
    answer = answer.replace("\\dfrac", "frac")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\,", "")
    return answer.lower()


def check_answer_correct(solution: str, ground_truth: str) -> bool:
    """Check if a solution's boxed answer matches ground truth."""
    extracted = extract_boxed_answer(solution)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(ground_truth)


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class APIClient:
    """Async OpenAI-compatible API client with concurrency control."""

    def __init__(
        self,
        endpoint: str = "http://10.254.138.189:8104",
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        max_concurrent: int = 64,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.completions_url = f"{self.endpoint}/v1/chat/completions"
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._stats = {"total": 0, "success": 0, "failed": 0, "retries": 0}

    async def _single_request(
        self,
        session: aiohttp.ClientSession,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """Send a single chat completion request with retries."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    self._stats["total"] += 1
                    async with session.post(
                        self.completions_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            text = data["choices"][0]["message"]["content"]
                            self._stats["success"] += 1
                            return text
                        elif resp.status == 429:
                            # Rate limited — back off
                            self._stats["retries"] += 1
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        else:
                            body = await resp.text()
                            logger.warning(
                                f"API error {resp.status}: {body[:200]}"
                            )
                            self._stats["retries"] += 1
                            await asyncio.sleep(self.retry_delay)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request error (attempt {attempt+1}): {e}")
                self._stats["retries"] += 1
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        self._stats["failed"] += 1
        return None

    async def generate_batch(
        self,
        all_messages: List[List[dict]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> List[Optional[str]]:
        """Send a batch of requests concurrently."""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._single_request(session, msgs, temperature, max_tokens)
                for msgs in all_messages
            ]
            results = await asyncio.gather(*tasks)
        return list(results)

    def get_stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_hint_messages(question: str, answer: str) -> List[dict]:
    """Build hint-based prompt (answer provided)."""
    return [
        {"role": "system", "content": HINT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"**Problem:** {question}\n\n"
            f"**Correct Answer:** {answer}\n\n"
            "Write a detailed step-by-step solution that arrives at this answer"
        )},
    ]


def build_free_messages(question: str) -> List[dict]:
    """Build free generation prompt (no answer given)."""
    return [
        {"role": "system", "content": FREE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Main Generation Logic
# ---------------------------------------------------------------------------

async def generate_hint_solutions(
    client: APIClient,
    questions: List[str],
    answers: List[str],
    num_solutions: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> List[List[str]]:
    """Generate solutions with hint (answer given). All kept."""
    all_messages = []
    index_map = []  # (question_idx, solution_idx)

    for i, (q, a) in enumerate(zip(questions, answers)):
        for s in range(num_solutions):
            all_messages.append(build_hint_messages(q, a))
            index_map.append((i, s))

    logger.info(f"Sending {len(all_messages)} hint requests...")
    results = await client.generate_batch(all_messages, temperature, max_tokens)

    # Collect
    solutions = [[] for _ in questions]
    for (qi, si), text in zip(index_map, results):
        if text is not None:
            solutions[qi].append(text)

    return solutions


async def generate_free_solutions(
    client: APIClient,
    questions: List[str],
    ground_truths: List[str],
    num_solutions: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> List[List[str]]:
    """Generate solutions freely, keep only correct ones."""
    all_messages = []
    index_map = []

    for i, q in enumerate(questions):
        for s in range(num_solutions):
            all_messages.append(build_free_messages(q))
            index_map.append((i, s))

    logger.info(f"Sending {len(all_messages)} free requests...")
    results = await client.generate_batch(all_messages, temperature, max_tokens)

    # Filter: keep only correct
    solutions = [[] for _ in questions]
    total_generated = 0
    total_correct = 0

    for (qi, si), text in zip(index_map, results):
        if text is None:
            continue
        total_generated += 1
        if check_answer_correct(text, ground_truths[qi]):
            solutions[qi].append(text)
            total_correct += 1

    logger.info(
        f"Free generation: {total_correct}/{total_generated} correct "
        f"({100*total_correct/max(total_generated,1):.1f}%)"
    )
    return solutions


async def run_generation(args):
    """Main async generation pipeline."""
    # Load dataset
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} questions from {args.input}")

    questions = [extract_question_from_prompt(r) for r in df["prompt"]]
    ground_truths = [extract_ground_truth(r) for r in df["reward_model"]]
    data_sources = list(df["data_source"])

    # Load existing to skip
    existing_ids = set()
    if args.append and os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["question_id"])
        print(f"Found {len(existing_ids)} existing questions, skipping")

    # Filter to new questions
    new_indices = []
    for i in range(len(questions)):
        qid = make_question_id(questions[i])
        if qid not in existing_ids:
            new_indices.append(i)

    if not new_indices:
        print("All questions already processed")
        return

    print(f"{len(new_indices)} new questions to process")

    # Create client
    client = APIClient(
        endpoint=args.endpoint,
        model=args.model,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mode = "a" if args.append else "w"

    # Process in batches
    total_written = 0
    t_start = time.time()

    for batch_start in range(0, len(new_indices), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(new_indices))
        batch_idx = new_indices[batch_start:batch_end]

        batch_questions = [questions[i] for i in batch_idx]
        batch_answers = [ground_truths[i] for i in batch_idx]

        print(f"\nBatch {batch_start//args.batch_size + 1}: "
              f"{len(batch_idx)} questions...")

        if args.strategy == "hint":
            batch_solutions = await generate_hint_solutions(
                client, batch_questions, batch_answers,
                num_solutions=args.num_solutions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        elif args.strategy == "free":
            batch_solutions = await generate_free_solutions(
                client, batch_questions, batch_answers,
                num_solutions=args.num_solutions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        elif args.strategy == "both":
            # Run both: hint for guaranteed solutions, free for diversity
            hint_sols = await generate_hint_solutions(
                client, batch_questions, batch_answers,
                num_solutions=max(1, args.num_solutions // 2),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            free_sols = await generate_free_solutions(
                client, batch_questions, batch_answers,
                num_solutions=args.num_solutions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            # Merge
            batch_solutions = [
                h + f for h, f in zip(hint_sols, free_sols)
            ]
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        # Write results
        with open(args.output, mode) as f:
            for j, idx in enumerate(batch_idx):
                sols = batch_solutions[j]
                if not sols:
                    continue
                record = {
                    "question_id": make_question_id(questions[idx]),
                    "question": questions[idx],
                    "ground_truth": ground_truths[idx],
                    "data_source": data_sources[idx],
                    "solutions": sols,
                    "strategy": args.strategy,
                    "model": args.model,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

        mode = "a"
        elapsed = time.time() - t_start
        rate = total_written / max(elapsed, 1)
        print(f"  Written: {total_written} questions ({rate:.1f} q/s)")

    stats = client.get_stats()
    elapsed = time.time() - t_start
    print(f"\nDone! {total_written} questions -> {args.output}")
    print(f"Time: {elapsed:.0f}s")
    print(f"API stats: {stats}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate reference solutions via API"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file (verl format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for solutions")
    parser.add_argument("--endpoint", type=str,
                        default="http://10.254.138.189:8104",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model name for API requests")
    parser.add_argument("--strategy", type=str, default="hint",
                        choices=["hint", "free", "both"],
                        help="hint: answer given; free: verify correctness; "
                             "both: mix of hint + free")
    parser.add_argument("--num_solutions", type=int, default=4,
                        help="Solutions to generate per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_concurrent", type=int, default=64,
                        help="Max concurrent API requests")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per request on failure")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Request timeout in seconds")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Process questions in batches of this size")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
