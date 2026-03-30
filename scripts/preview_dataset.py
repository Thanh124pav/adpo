#!/usr/bin/env python3
"""
Preview datasets: show statistics, level/category distribution, and sample rows.

Works with both processed .parquet files and raw HuggingFace dataset names.

Usage:
    # Preview a processed parquet file
    python scripts/preview_dataset.py data/processed/train/math.parquet

    # Preview a raw HuggingFace dataset (before processing)
    python scripts/preview_dataset.py --hf lighteval/MATH --split train

    # Show level distribution only
    python scripts/preview_dataset.py data/processed/train/math.parquet --stats-only

    # Preview with level filter
    python scripts/preview_dataset.py data/processed/train/math.parquet --level "Level 4,Level 5"

    # Show more samples
    python scripts/preview_dataset.py data/processed/train/math.parquet -k 10

    # Preview all parquet files in a directory
    python scripts/preview_dataset.py data/processed/train/
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate(text, max_len=120):
    text = str(text).replace("\n", " ")
    return text[:max_len] + "…" if len(text) > max_len else text


def print_header(title, width=80):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_distribution(counter, title, top_n=20):
    """Print a frequency table from a Counter."""
    if not counter:
        return
    total = sum(counter.values())
    print(f"\n  {title} ({len(counter)} unique values, {total} total):")
    print(f"  {'Value':<40} {'Count':>8} {'Pct':>7}")
    print(f"  {'-'*40} {'-'*8} {'-'*7}")
    for val, cnt in counter.most_common(top_n):
        pct = cnt / total * 100
        print(f"  {truncate(str(val), 40):<40} {cnt:>8} {pct:>6.1f}%")
    if len(counter) > top_n:
        rest = sum(cnt for _, cnt in counter.most_common()[top_n:])
        print(f"  {'... (others)':<40} {rest:>8} {rest/total*100:>6.1f}%")


# ---------------------------------------------------------------------------
# Parquet preview
# ---------------------------------------------------------------------------

def extract_field(value, key):
    """Extract a nested field from a dict or JSON string."""
    if isinstance(value, dict):
        return value.get(key)
    if isinstance(value, str):
        try:
            return json.loads(value).get(key)
        except (json.JSONDecodeError, AttributeError):
            return None
    return None


def preview_parquet(path, num_samples=5, level_filter=None, stats_only=False, seed=42):
    df = pd.read_parquet(path)
    print_header(f"{path}  ({len(df)} rows)")

    # --- Basic stats ---
    print(f"\n  Columns: {list(df.columns)}")
    print(f"  Rows:    {len(df)}")

    # --- Column types (check first row) ---
    if len(df) > 0:
        row0 = df.iloc[0]
        types = {col: type(row0[col]).__name__ for col in df.columns}
        print(f"  Types:   {types}")

    # --- data_source distribution ---
    if "data_source" in df.columns:
        print_distribution(Counter(df["data_source"]), "data_source distribution")

    # --- Level / category distribution from extra_info ---
    level_counter = Counter()
    type_counter = Counter()
    source_counter = Counter()
    for val in df.get("extra_info", []):
        level = extract_field(val, "level")
        typ = extract_field(val, "type")
        source = extract_field(val, "source") or extract_field(val, "problem_source")
        if level:
            level_counter[level] += 1
        if typ:
            type_counter[typ] += 1
        if source:
            source_counter[source] += 1

    if level_counter:
        print_distribution(level_counter, "Level distribution")
    if type_counter:
        print_distribution(type_counter, "Type/Subject distribution")
    if source_counter:
        print_distribution(source_counter, "Source distribution")

    if not level_counter and not type_counter and not source_counter:
        print("\n  (No level/type/source metadata found in extra_info)")

    # --- Apply level filter ---
    if level_filter:
        allowed = {v.strip().lower() for v in level_filter.split(",") if v.strip()}
        mask = df["extra_info"].apply(
            lambda v: (
                str(extract_field(v, "level") or "").strip().lower() in allowed
                or str(extract_field(v, "source") or extract_field(v, "problem_source") or "").strip().lower() in allowed
            )
        )
        df = df[mask]
        print(f"\n  After --level filter '{level_filter}': {len(df)} rows remaining")

    if stats_only:
        return

    # --- Sample rows ---
    k = min(num_samples, len(df))
    if k == 0:
        print("\n  No rows to sample.")
        return

    sample = df.sample(n=k, random_state=seed)
    print(f"\n  --- {k} Sample Rows ---")

    for idx, (_, row) in enumerate(sample.iterrows()):
        print(f"\n  [{idx + 1}/{k}]")
        # prompt: ndarray / list / str — extract user message
        prompt = row.get("prompt")
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            user_msg = next((m["content"] for m in prompt if isinstance(m, dict) and m.get("role") == "user"), str(prompt))
        elif isinstance(prompt, str):
            try:
                msgs = json.loads(prompt)
                user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), prompt)
            except (json.JSONDecodeError, TypeError):
                user_msg = prompt
        else:
            user_msg = str(prompt)
        print(f"    Question:     {truncate(user_msg, 200)}")

        # ground truth
        rm = row.get("reward_model")
        gt = extract_field(rm, "ground_truth") if rm else None
        if gt:
            print(f"    Ground Truth: {truncate(str(gt), 200)}")

        # extra_info highlights
        extra = row.get("extra_info")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        if isinstance(extra, dict):
            tags = []
            for key in ("level", "type", "source", "problem_source", "config"):
                if extra.get(key):
                    tags.append(f"{key}={extra[key]}")
            if tags:
                print(f"    Metadata:     {', '.join(tags)}")


# ---------------------------------------------------------------------------
# HuggingFace dataset preview
# ---------------------------------------------------------------------------

def preview_hf(dataset_name, split="train", num_samples=5, max_rows=1000):
    import datasets as hf_datasets

    print_header(f"HuggingFace: {dataset_name} (split={split})")

    try:
        ds = hf_datasets.load_dataset(dataset_name, split=split, streaming=True)
    except Exception:
        # Try with common configs
        for config in ("main", "all", "default"):
            try:
                ds = hf_datasets.load_dataset(dataset_name, config, split=split, streaming=True)
                print(f"  (loaded with config='{config}')")
                break
            except Exception:
                continue
        else:
            print(f"  ERROR: Could not load {dataset_name}")
            return

    # Collect rows for stats
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i >= max_rows - 1:
            break

    print(f"\n  Scanned: {len(rows)} rows (of split '{split}')")
    print(f"  Columns: {list(rows[0].keys()) if rows else '(empty)'}")

    # Detect and show distribution for common metadata fields
    for field in ("level", "type", "subject", "source", "problem_source", "difficulty", "category"):
        vals = [r.get(field) for r in rows if r.get(field)]
        if vals:
            print_distribution(Counter(vals), f"'{field}' distribution")

    # Sample
    import random
    random.seed(42)
    samples = random.sample(rows, min(num_samples, len(rows)))
    print(f"\n  --- {len(samples)} Sample Rows ---")
    for idx, row in enumerate(samples):
        print(f"\n  [{idx + 1}/{len(samples)}]")
        # Try common question/answer fields
        q = row.get("problem") or row.get("question") or row.get("input", "")
        a = row.get("solution") or row.get("answer") or row.get("expected_answer", "")
        print(f"    Question: {truncate(str(q), 200)}")
        if a:
            print(f"    Answer:   {truncate(str(a), 200)}")
        meta = {k: v for k, v in row.items()
                if k not in ("problem", "question", "input", "solution", "answer", "expected_answer")
                and v}
        if meta:
            print(f"    Fields:   {truncate(str(meta), 200)}")


# ---------------------------------------------------------------------------
# Directory preview
# ---------------------------------------------------------------------------

def preview_directory(dir_path, num_samples=3, level_filter=None, stats_only=False):
    parquet_files = sorted(Path(dir_path).rglob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {dir_path}")
        return

    print_header(f"Directory: {dir_path}  ({len(parquet_files)} parquet files)")
    for f in parquet_files:
        rel = f.relative_to(dir_path)
        size_mb = f.stat().st_size / (1024 * 1024)
        df = pd.read_parquet(f)
        print(f"  {str(rel):<45} {len(df):>8} rows  ({size_mb:.1f} MB)")

    for f in parquet_files:
        preview_parquet(str(f), num_samples=num_samples, level_filter=level_filter, stats_only=stats_only)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preview datasets: stats, level distribution, and samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=None,
                        help="Path to .parquet file or directory of parquet files")
    parser.add_argument("--hf", type=str, default=None,
                        help="HuggingFace dataset name (e.g. 'lighteval/MATH')")
    parser.add_argument("--split", type=str, default="train",
                        help="HF dataset split (default: train)")
    parser.add_argument("-k", "--num-samples", type=int, default=5,
                        help="Number of sample rows to display (default: 5)")
    parser.add_argument("--level", type=str, default=None,
                        help="Filter by level/source (comma-separated)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Show only statistics, no sample rows")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    if not args.path and not args.hf:
        # Default: preview all processed data
        processed_dir = "data/processed"
        if os.path.isdir(processed_dir):
            preview_directory(processed_dir, args.num_samples, args.level, args.stats_only)
        else:
            parser.print_help()
        return

    if args.hf:
        preview_hf(args.hf, split=args.split, num_samples=args.num_samples)
        return

    path = args.path
    if os.path.isdir(path):
        preview_directory(path, args.num_samples, args.level, args.stats_only)
    elif os.path.isfile(path):
        preview_parquet(path, args.num_samples, args.level, args.stats_only, args.seed)
    else:
        print(f"Error: '{path}' not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
