#!/usr/bin/env python3
"""Sample random rows from a .parquet file and print/save them."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample random rows from a .parquet file")
    parser.add_argument("file", help="Path to the .parquet file")
    parser.add_argument("-k", "--num-samples", type=int, default=5, help="Number of random samples (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--columns", nargs="+", default=None, help="Only show these columns")
    parser.add_argument("-o", "--output", default=None, help="Output .parquet file path")
    parser.add_argument("--max-width", type=int, default=None, help="Max column width (default: no limit)")
    args = parser.parse_args()

    df = pd.read_parquet(args.file, columns=args.columns)
    k = min(args.num_samples, len(df))
    sample = df.sample(n=k, random_state=args.seed)

    if args.output:
        sample.to_parquet(args.output)
        print(f"Saved {k}/{len(df)} rows to {args.output}")
    else:
        pd.set_option("display.max_colwidth", args.max_width)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_rows", None)
        for i, (idx, row) in enumerate(sample.iterrows()):
            print(f"{'='*80}\nRow {i} (index={idx})\n{'='*80}")
            for col in row.index:
                val = str(row[col])
                print(f"  {col}: {val}")
            print()


if __name__ == "__main__":
    main()
