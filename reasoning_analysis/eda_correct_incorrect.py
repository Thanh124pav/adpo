"""
EDA: Compare statistics between correct and incorrect responses.

Loads the JSONL output from evaluate.py (which now includes `correct` and `score`
fields) and computes per-group (correct / incorrect) statistics:
  - mean, std of: neg_log_prob_mean, entropy_mean, response_length
  - covariance matrix of the three features
  - statistical significance tests (t-test)

Usage:
    python reasoning_analysis/eda_correct_incorrect.py \
        --input_path reasoning_analysis/outputs/analysis_math500.jsonl \
        --output_dir reasoning_analysis/outputs/eda
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_results(path: str) -> list[dict]:
    """Load JSONL or JSON results."""
    results = []
    if path.endswith(".json"):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def build_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with per-response summary features."""
    rows = []
    for r in results:
        tokens = r.get("tokens", [])
        if not tokens:
            continue

        neg_lps = [t["neg_log_prob"] for t in tokens]
        entropies = [t["entropy"] for t in tokens]

        rows.append({
            "prompt_idx": r["prompt_idx"],
            "sample_idx": r.get("sample_idx", 0),
            "correct": r.get("correct", False),
            "score": r.get("score", 0.0),
            "data_source": r.get("data_source", "unknown"),
            "response_length": len(tokens),
            "neg_log_prob_mean": float(np.mean(neg_lps)),
            "neg_log_prob_std": float(np.std(neg_lps)),
            "neg_log_prob_median": float(np.median(neg_lps)),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "entropy_median": float(np.median(entropies)),
        })
    return pd.DataFrame(rows)


def print_group_stats(df: pd.DataFrame, label: str):
    """Print mean, std for key features of a group."""
    features = ["neg_log_prob_mean", "entropy_mean", "response_length"]
    print(f"\n--- {label} (n={len(df)}) ---")
    for feat in features:
        vals = df[feat]
        print(f"  {feat:25s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={vals.median():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    # Covariance matrix
    if len(df) > 1:
        cov = df[features].cov()
        corr = df[features].corr()
        print(f"\n  Covariance matrix:")
        print(cov.to_string(float_format=lambda x: f"{x:.6f}"))
        print(f"\n  Correlation matrix:")
        print(corr.to_string(float_format=lambda x: f"{x:.4f}"))


def run_ttest(df_correct: pd.DataFrame, df_incorrect: pd.DataFrame):
    """Run Welch's t-test between correct and incorrect groups."""
    from scipy import stats

    features = ["neg_log_prob_mean", "entropy_mean", "response_length"]
    print("\n--- Welch's t-test (correct vs incorrect) ---")
    for feat in features:
        if len(df_correct) < 2 or len(df_incorrect) < 2:
            print(f"  {feat}: not enough samples for t-test")
            continue
        t_stat, p_val = stats.ttest_ind(
            df_correct[feat], df_incorrect[feat], equal_var=False
        )
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {feat:25s}  t={t_stat:+.4f}  p={p_val:.6f} {sig}")


def save_plots(df: pd.DataFrame, output_dir: str):
    """Generate and save comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    features = ["neg_log_prob_mean", "entropy_mean", "response_length"]
    labels = {True: "Correct", False: "Incorrect"}
    colors = {True: "#2ecc71", False: "#e74c3c"}

    df_correct = df[df["correct"]]
    df_incorrect = df[~df["correct"]]

    # 1. Box plots for each feature
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, features):
        data = [df_correct[feat].values, df_incorrect[feat].values]
        bp = ax.boxplot(data, labels=["Correct", "Incorrect"], patch_artist=True)
        bp["boxes"][0].set_facecolor(colors[True])
        bp["boxes"][1].set_facecolor(colors[False])
        ax.set_title(feat)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Feature Distributions: Correct vs Incorrect", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplots.png"), dpi=150)
    plt.close(fig)

    # 2. Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, features):
        bins = 30
        ax.hist(df_correct[feat], bins=bins, alpha=0.6, color=colors[True],
                label="Correct", density=True)
        ax.hist(df_incorrect[feat], bins=bins, alpha=0.6, color=colors[False],
                label="Incorrect", density=True)
        ax.set_title(feat)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Feature Histograms: Correct vs Incorrect", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "histograms.png"), dpi=150)
    plt.close(fig)

    # 3. Scatter: neg_log_prob_mean vs entropy_mean, colored by correctness
    fig, ax = plt.subplots(figsize=(8, 6))
    for correct_val, group in df.groupby("correct"):
        ax.scatter(group["neg_log_prob_mean"], group["entropy_mean"],
                   alpha=0.5, s=20, color=colors[correct_val],
                   label=labels[correct_val])
    ax.set_xlabel("neg_log_prob_mean")
    ax.set_ylabel("entropy_mean")
    ax.set_title("neg_log_prob vs entropy (colored by correctness)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_nlp_entropy.png"), dpi=150)
    plt.close(fig)

    # 4. Scatter: response_length vs neg_log_prob_mean
    fig, ax = plt.subplots(figsize=(8, 6))
    for correct_val, group in df.groupby("correct"):
        ax.scatter(group["response_length"], group["neg_log_prob_mean"],
                   alpha=0.5, s=20, color=colors[correct_val],
                   label=labels[correct_val])
    ax.set_xlabel("response_length (tokens)")
    ax.set_ylabel("neg_log_prob_mean")
    ax.set_title("Response Length vs neg_log_prob (colored by correctness)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_length_nlp.png"), dpi=150)
    plt.close(fig)

    # 5. Scatter: response_length vs entropy_mean
    fig, ax = plt.subplots(figsize=(8, 6))
    for correct_val, group in df.groupby("correct"):
        ax.scatter(group["response_length"], group["entropy_mean"],
                   alpha=0.5, s=20, color=colors[correct_val],
                   label=labels[correct_val])
    ax.set_xlabel("response_length (tokens)")
    ax.set_ylabel("entropy_mean")
    ax.set_title("Response Length vs Entropy (colored by correctness)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_length_entropy.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="EDA: Compare correct vs incorrect responses"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to JSONL/JSON output from evaluate.py")
    parser.add_argument("--output_dir", type=str,
                        default="reasoning_analysis/outputs/eda",
                        help="Directory to save plots and CSV")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots (useful on headless servers)")
    args = parser.parse_args()

    # Load
    print(f"Loading results from {args.input_path} ...")
    results = load_results(args.input_path)
    print(f"Loaded {len(results)} responses")

    # Check that accuracy info is present
    has_correct = any("correct" in r for r in results)
    if not has_correct:
        print("WARNING: 'correct' field not found in results. "
              "Re-run evaluate.py to generate results with accuracy info.")
        return

    # Build DataFrame
    df = build_dataframe(results)
    print(f"Built DataFrame with {len(df)} rows")

    df_correct = df[df["correct"]]
    df_incorrect = df[~df["correct"]]

    # Overall accuracy
    n_correct = len(df_correct)
    n_total = len(df)
    print(f"\n{'='*60}")
    print("EDA: CORRECT vs INCORRECT")
    print(f"{'='*60}")
    print(f"Accuracy: {n_correct}/{n_total} = {n_correct/n_total:.4f} "
          f"({n_correct/n_total*100:.1f}%)")

    # Per-group stats
    print_group_stats(df_correct, "CORRECT")
    print_group_stats(df_incorrect, "INCORRECT")

    # T-test
    try:
        run_ttest(df_correct, df_incorrect)
    except ImportError:
        print("\n(scipy not installed, skipping t-test)")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to {csv_path}")

    # Plots
    if not args.no_plots:
        try:
            save_plots(df, args.output_dir)
        except ImportError as e:
            print(f"\n(matplotlib not available, skipping plots: {e})")


if __name__ == "__main__":
    main()
