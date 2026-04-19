"""
Convert adpo parquet datasets → SPO HuggingFace Dataset format.

SPO expects a DatasetDict saved to disk via save_to_disk(), with splits
"train" and "test". Required columns depend on the task type:

  MATH task  → columns: problem, solution, answer, _treetune__idx (auto)
  GSM8K task → columns: question, answer, _treetune__idx (auto)

Usage:
    python spo/scripts/convert_adpo_to_spo.py \
        --input  data/processed/train/math.parquet \
        --test   data/processed/test/math.parquet \
        --output spo/data/math \
        --task   math

    python spo/scripts/convert_adpo_to_spo.py \
        --input  data/processed/train/gsm8k.parquet \
        --test   data/processed/test/gsm8k.parquet \
        --output spo/data/gsm8k \
        --task   gsm8k
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict


def extract_question(prompt) -> str:
    """Extract plain question text from verl prompt (list of chat messages or str)."""
    if isinstance(prompt, str):
        try:
            messages = json.loads(prompt)
        except (json.JSONDecodeError, TypeError):
            return prompt
    elif isinstance(prompt, list):
        messages = prompt
    else:
        return str(prompt)

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    # fallback: last message content
    if messages:
        return messages[-1].get("content", str(messages[-1]))
    return str(prompt)


def extract_ground_truth(reward_model) -> str:
    """Extract ground truth answer from verl reward_model field."""
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", "")
    if isinstance(reward_model, str):
        try:
            d = json.loads(reward_model)
            return d.get("ground_truth", reward_model)
        except (json.JSONDecodeError, TypeError):
            return reward_model
    return str(reward_model)


def convert_to_math(df: pd.DataFrame) -> list:
    """Convert to MATH task format: columns problem, solution, answer."""
    records = []
    for _, row in df.iterrows():
        question = extract_question(row["prompt"])
        ground_truth = extract_ground_truth(row["reward_model"])
        records.append({
            "problem": question,
            "solution": ground_truth,   # SPO MATH task reads solution col
            "answer": ground_truth,     # some subtasks also read answer col
        })
    return records


def convert_to_gsm8k(df: pd.DataFrame) -> list:
    """Convert to GSM8K task format: columns question, answer."""
    records = []
    for _, row in df.iterrows():
        question = extract_question(row["prompt"])
        ground_truth = extract_ground_truth(row["reward_model"])
        # GSM8K original format expects answer with #### delimiter
        # If the ground truth is already in that format, keep it;
        # otherwise wrap it so extract_gold_answer_from_text() works.
        if "####" not in str(ground_truth):
            answer = f"...\n#### {ground_truth}"
        else:
            answer = ground_truth
        records.append({
            "question": question,
            "answer": answer,
        })
    return records


CONVERTERS = {
    "math": convert_to_math,
    "gsm8k": convert_to_gsm8k,
}


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_split(path: str, task: str) -> Dataset:
    df = load_parquet(path)
    converter = CONVERTERS[task]
    records = converter(df)
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Train parquet file")
    parser.add_argument("--test",   default=None,  help="Test parquet file (optional)")
    parser.add_argument("--output", required=True, help="Output directory for SPO dataset")
    parser.add_argument("--task",   required=True, choices=list(CONVERTERS.keys()),
                        help="SPO task type")
    args = parser.parse_args()

    print(f"Converting {args.input} → {args.output} (task={args.task})")

    splits = {"train": build_split(args.input, args.task)}

    if args.test:
        print(f"  + test split from {args.test}")
        splits["test"] = build_split(args.test, args.task)

    ds = DatasetDict(splits)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    print(f"Saved to {out}")
    print(f"  train: {len(splits['train'])} rows, columns: {splits['train'].column_names}")
    if "test" in splits:
        print(f"  test:  {len(splits['test'])} rows")


if __name__ == "__main__":
    main()
