"""
Shared helpers for run_tree_analysis.py and run_local_hf.py.
"""
import datetime
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List


# Patterns appended by MCQ/benchmark datasets that confuse open-ended generation
_MCQ_NOISE = re.compile(
    r'\s*[\(\*]*\s*(?:Please\s+answer\s+with\s+an?\s+option|'
    r'Your\s+answer\s+should\s+be\s+correct|'
    r'answer\s+with\s+one\s+of\s+the\s+following|'
    r'\(A\)\s*[\w\s]+\(B\)|'
    r'Options?:\s*\(A\)).*$',
    re.IGNORECASE | re.DOTALL,
)


def load_parquet_examples(path: str, n: int, seed: int = 42) -> List[Dict]:
    """Load *n* random examples from a verl-format parquet file.

    Schema expected (from data/prepare_datasets.py):
        prompt        – np.ndarray of {"role":…,"content":…} dicts
        reward_model  – dict with key "ground_truth"
        data_source   – str
    """
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

        question = _MCQ_NOISE.sub("", question).strip()

        rm = row["reward_model"]
        gold_answer = str(rm.get("ground_truth", "")) if isinstance(rm, dict) else str(rm)

        examples.append({
            "question":    question,
            "gold_answer": gold_answer,
            "source":      str(row.get("data_source", "parquet")),
        })

    print(f"[parquet] Loaded {len(examples)} examples from {path}  (seed={seed})")
    return examples


def make_stem(model_name: str, tree_label: str, question: str) -> str:
    """Unique, stable filename stem per (model, tree, question).

    8-char SHA-256 prefix → content-addressed: same question = same stem,
    different questions never collide (~1-in-4B per model+tree pair).
    """
    q8 = hashlib.sha256(question.encode()).hexdigest()[:8]
    return f"{model_name.split('/')[-1]}_{tree_label}_{q8}"


def append_summary(save_dir: Path, record: dict, stem: str) -> None:
    """Append one lightweight result line to save_dir/summary.jsonl.

    Append-only JSONL: accumulates across runs without overwriting.
    The full tree lives in the per-question .json file; the summary keeps
    only scalar fields for easy aggregation with pandas / jq.
    """
    line = {
        "model":         record["model"],
        "tree_config":   record["tree_config"],
        "question_hash": stem.rsplit("_", 1)[-1],   # reuse hash from stem
        "source":        record.get("source", ""),
        "question":      record["question"],
        "gold_answer":   record["gold_answer"],
        "elapsed_s":     record["elapsed_s"],
        "root_V":        record["root_V"],
        "root_P":        record["root_P"],
        "timestamp":     datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    summary_path = save_dir / "summary.jsonl"
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")
    print(f"  SUMM  → {summary_path}")
