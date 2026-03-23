#!/usr/bin/env python3
"""Sample random rows from a .parquet file and print them."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample random rows from a .parquet file")
    parser.add_argument("file", help="Path to the .parquet file")
    parser.add_argument("-k", "--num-samples", type=int, default=5, help="Number of random samples (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--columns", nargs="+", default=None, help="Only show these columns")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of table")
    args = parser.parse_args()

    df = pd.read_parquet(args.file, columns=args.columns)
    k = min(args.num_samples, len(df))
    sample = df.sample(n=k, random_state=args.seed)

    print(f"Total rows: {len(df)} | Sampling: {k}\n")

    if args.json:
        print(sample.to_json(orient="records", indent=2, force_ascii=False))
    else:
        pd.set_option("display.max_colwidth", 120)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(sample.to_string())


if __name__ == "__main__":
    main()
