#!/usr/bin/env python3
"""Sample random rows from a .parquet file and save to a new .parquet file."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample random rows from a .parquet file")
    parser.add_argument("file", help="Path to the .parquet file")
    parser.add_argument("-k", "--num-samples", type=int, default=5, help="Number of random samples (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--columns", nargs="+", default=None, help="Only show these columns")
    parser.add_argument("-o", "--output", required=True, help="Output .parquet file path")
    args = parser.parse_args()

    df = pd.read_parquet(args.file, columns=args.columns)
    k = min(args.num_samples, len(df))
    sample = df.sample(n=k, random_state=args.seed)
    sample.to_parquet(args.output)

    print(f"Saved {k}/{len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
