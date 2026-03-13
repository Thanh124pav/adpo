"""
Generate multiple reference solutions per question using a teacher model.

Two strategies:
1. "teacher": Use a large model (e.g., Qwen2.5-72B) to generate solutions
2. "hint": Give the training model the correct answer and ask it to produce
   a step-by-step solution leading to that answer

Usage:
    # Strategy 1: Teacher model generates solutions
    python data/generate_solutions.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions.jsonl \
        --model Qwen/Qwen2.5-72B-Instruct \
        --strategy teacher \
        --num_solutions 4

    # Strategy 2: Training model + hint (answer given)
    python data/generate_solutions.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions.jsonl \
        --model Qwen/Qwen2.5-Math-7B \
        --strategy hint \
        --num_solutions 4

    # Generate for all training datasets
    bash scripts/generate_all_solutions.sh
"""

import argparse
import json
import os
import hashlib
from typing import List, Dict

import pandas as pd


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

HINT_SYSTEM_PROMPT = (
    "You are given a math problem and its correct answer. "
    "Write a detailed step-by-step solution that arrives at the given answer. "
    "Put your final answer within \\boxed{}."
)


def make_question_id(question: str) -> str:
    """Deterministic ID for a question (for deduplication)."""
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


def build_teacher_prompts(questions: List[str]) -> List[List[dict]]:
    """Build prompts for teacher model (no hint)."""
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]


def build_hint_prompts(
    questions: List[str], answers: List[str]
) -> List[List[dict]]:
    """Build prompts with answer hint."""
    prompts = []
    for q, a in zip(questions, answers):
        prompts.append([
            {"role": "system", "content": HINT_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"**Problem:** {q}\n\n"
                f"**Correct Answer:** {a}\n\n"
                f"Write a detailed step-by-step solution."
            )},
        ])
    return prompts


def generate_with_vllm(
    model_path: str,
    prompts: List[List[dict]],
    num_solutions: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
) -> List[List[str]]:
    """Generate multiple solutions per prompt using vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_solutions,
        top_p=0.95,
    )

    # Format prompts
    formatted = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    outputs = llm.generate(formatted, sampling_params)

    results = []
    for output in outputs:
        solutions = [o.text for o in output.outputs]
        results.append(solutions)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate reference solutions")
    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file (verl format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for solutions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct",
                        help="Model to generate solutions")
    parser.add_argument("--strategy", type=str, default="teacher",
                        choices=["teacher", "hint"],
                        help="teacher: model solves freely; hint: model given answer")
    parser.add_argument("--num_solutions", type=int, default=4,
                        help="Number of solutions to generate per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Process questions in batches")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} questions from {args.input}")

    # Extract questions and answers
    questions = [extract_question_from_prompt(r) for r in df["prompt"]]
    ground_truths = [extract_ground_truth(r) for r in df["reward_model"]]
    data_sources = list(df["data_source"])

    # Load existing solutions to skip already-done questions
    existing_ids = set()
    if args.append and os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["question_id"])
        print(f"Found {len(existing_ids)} existing questions, skipping")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mode = "a" if args.append else "w"

    # Process in batches
    total = len(questions)
    written = 0

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)

        batch_questions = []
        batch_answers = []
        batch_indices = []

        for i in range(batch_start, batch_end):
            qid = make_question_id(questions[i])
            if qid in existing_ids:
                continue
            batch_questions.append(questions[i])
            batch_answers.append(ground_truths[i])
            batch_indices.append(i)

        if not batch_questions:
            continue

        print(f"Generating solutions for batch {batch_start}-{batch_end} "
              f"({len(batch_questions)} new questions)...")

        # Build prompts
        if args.strategy == "teacher":
            prompts = build_teacher_prompts(batch_questions)
        else:
            prompts = build_hint_prompts(batch_questions, batch_answers)

        # Generate
        all_solutions = generate_with_vllm(
            model_path=args.model,
            prompts=prompts,
            num_solutions=args.num_solutions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # Save
        with open(args.output, mode) as f:
            for idx, solutions in zip(batch_indices, all_solutions):
                record = {
                    "question_id": make_question_id(questions[idx]),
                    "question": questions[idx],
                    "ground_truth": ground_truths[idx],
                    "data_source": data_sources[idx],
                    "solutions": solutions,
                    "strategy": args.strategy,
                    "model": args.model,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        mode = "a"  # Switch to append after first batch
        print(f"  Written {written} questions so far")

    print(f"\nDone! Generated solutions for {written} questions -> {args.output}")


if __name__ == "__main__":
    main()
