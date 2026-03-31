"""
Reasoning Analysis: Visualize per-token -log_prob and entropy.

Generates:
  1. Statistical plots (PNG): distributions, means, stds, per-position trends
  2. HTML file with -log_prob annotations above each token
  3. HTML file with entropy annotations above each token

Usage:
    python reasoning_analysis/visualize.py \
        --input_path reasoning_analysis/outputs/analysis.jsonl \
        --output_dir reasoning_analysis/outputs/visualizations
"""

import argparse
import json
import os
import html as html_lib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def _configure_matplotlib():
    """Disable mathtext parsing so token labels with '$' don't crash plots."""
    import matplotlib
    try:
        matplotlib.rcParams["text.parse_math"] = False
    except Exception:
        # Older matplotlib versions may not expose this rcParam.
        pass


def load_results(input_path: str) -> list:
    """Load analysis results from JSON or JSONL."""
    if input_path.endswith(".json"):
        with open(input_path, encoding="utf-8") as f:
            return json.load(f)
    else:
        results = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results


# ---------------------------------------------------------------------------
# Statistical Plots
# ---------------------------------------------------------------------------


def generate_statistics_plots(results: list, output_dir: str):
    """Generate statistical analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Collect all token-level data
    all_neg_lps = []
    all_entropies = []
    per_response_neg_lp_means = []
    per_response_entropy_means = []
    per_response_neg_lp_stds = []
    per_response_entropy_stds = []
    position_neg_lps = {}  # position -> list of values
    position_entropies = {}

    for r in results:
        neg_lps = []
        ents = []
        for t in r["tokens"]:
            nlp = t["neg_log_prob"]
            ent = t["entropy"]
            all_neg_lps.append(nlp)
            all_entropies.append(ent)
            neg_lps.append(nlp)
            ents.append(ent)
            pos = t["position"]
            position_neg_lps.setdefault(pos, []).append(nlp)
            position_entropies.setdefault(pos, []).append(ent)

        if neg_lps:
            per_response_neg_lp_means.append(np.mean(neg_lps))
            per_response_entropy_means.append(np.mean(ents))
            per_response_neg_lp_stds.append(np.std(neg_lps))
            per_response_entropy_stds.append(np.std(ents))

    all_neg_lps = np.array(all_neg_lps)
    all_entropies = np.array(all_entropies)

    # --- Plot 1: Distribution of -log_prob ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(all_neg_lps, bins=100, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0].axvline(np.mean(all_neg_lps), color="red", linestyle="--", label=f"Mean={np.mean(all_neg_lps):.3f}")
    axes[0].axvline(np.median(all_neg_lps), color="orange", linestyle="--", label=f"Median={np.median(all_neg_lps):.3f}")
    axes[0].set_xlabel("-log P(token)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of -log_prob (all tokens)")
    axes[0].legend()

    axes[1].hist(all_entropies, bins=100, color="coral", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1].axvline(np.mean(all_entropies), color="red", linestyle="--", label=f"Mean={np.mean(all_entropies):.3f}")
    axes[1].axvline(np.median(all_entropies), color="orange", linestyle="--", label=f"Median={np.median(all_entropies):.3f}")
    axes[1].set_xlabel("Entropy H(token)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Entropy (all tokens)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_all_tokens.png"), dpi=150)
    plt.close()

    # --- Plot 2: Per-response mean & std ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(per_response_neg_lp_means, bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0, 0].set_xlabel("Mean -log_prob per response")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Per-Response Mean -log_prob\n(overall mean={np.mean(per_response_neg_lp_means):.3f})")

    axes[0, 1].hist(per_response_entropy_means, bins=50, color="coral", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0, 1].set_xlabel("Mean Entropy per response")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Per-Response Mean Entropy\n(overall mean={np.mean(per_response_entropy_means):.3f})")

    axes[1, 0].hist(per_response_neg_lp_stds, bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1, 0].set_xlabel("Std -log_prob per response")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Per-Response Std -log_prob\n(overall mean={np.mean(per_response_neg_lp_stds):.3f})")

    axes[1, 1].hist(per_response_entropy_stds, bins=50, color="coral", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1, 1].set_xlabel("Std Entropy per response")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"Per-Response Std Entropy\n(overall mean={np.mean(per_response_entropy_stds):.3f})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_response_stats.png"), dpi=150)
    plt.close()

    # --- Plot 3: Per-position mean (averaged over responses) ---
    max_pos = min(max(position_neg_lps.keys()) + 1, 512)  # Cap at 512 for readability
    positions = list(range(max_pos))
    pos_nlp_means = [np.mean(position_neg_lps.get(p, [0])) for p in positions]
    pos_nlp_stds = [np.std(position_neg_lps.get(p, [0])) for p in positions]
    pos_ent_means = [np.mean(position_entropies.get(p, [0])) for p in positions]
    pos_ent_stds = [np.std(position_entropies.get(p, [0])) for p in positions]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(positions, pos_nlp_means, color="steelblue", linewidth=0.8, label="Mean -log_prob")
    axes[0].fill_between(positions,
                         np.array(pos_nlp_means) - np.array(pos_nlp_stds),
                         np.array(pos_nlp_means) + np.array(pos_nlp_stds),
                         alpha=0.2, color="steelblue")
    axes[0].set_xlabel("Token Position")
    axes[0].set_ylabel("-log_prob")
    axes[0].set_title("Mean -log_prob by Token Position (across responses)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(positions, pos_ent_means, color="coral", linewidth=0.8, label="Mean Entropy")
    axes[1].fill_between(positions,
                         np.array(pos_ent_means) - np.array(pos_ent_stds),
                         np.array(pos_ent_means) + np.array(pos_ent_stds),
                         alpha=0.2, color="coral")
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Mean Entropy by Token Position (across responses)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_position_trends.png"), dpi=150)
    plt.close()

    # --- Plot 4: Box plots per-response ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_responses = results[:min(20, len(results))]  # Show first 20
    neg_lp_per_resp = [[t["neg_log_prob"] for t in r["tokens"]] for r in sample_responses if r["tokens"]]
    ent_per_resp = [[t["entropy"] for t in r["tokens"]] for r in sample_responses if r["tokens"]]

    if neg_lp_per_resp:
        axes[0].boxplot(neg_lp_per_resp, vert=True)
        axes[0].set_xlabel("Response Index")
        axes[0].set_ylabel("-log_prob")
        axes[0].set_title("Box Plot: -log_prob per Response (first 20)")

        axes[1].boxplot(ent_per_resp, vert=True)
        axes[1].set_xlabel("Response Index")
        axes[1].set_ylabel("Entropy")
        axes[1].set_title("Box Plot: Entropy per Response (first 20)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_per_response.png"), dpi=150)
    plt.close()

    # --- Plot 5: Per-token trajectory for individual responses ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, ax_row in enumerate(axes):
        if i >= len(results):
            break
        r = results[i]
        positions_r = [t["position"] for t in r["tokens"]]
        nlps_r = [t["neg_log_prob"] for t in r["tokens"]]
        ents_r = [t["entropy"] for t in r["tokens"]]

        ax_row[0].plot(positions_r, nlps_r, color="steelblue", linewidth=0.6)
        ax_row[0].set_xlabel("Token Position")
        ax_row[0].set_ylabel("-log_prob")
        ax_row[0].set_title(f"Response {i}: -log_prob Trajectory")
        ax_row[0].grid(True, alpha=0.3)

        ax_row[1].plot(positions_r, ents_r, color="coral", linewidth=0.6)
        ax_row[1].set_xlabel("Token Position")
        ax_row[1].set_ylabel("Entropy")
        ax_row[1].set_title(f"Response {i}: Entropy Trajectory")
        ax_row[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "individual_trajectories.png"), dpi=150)
    plt.close()

    # --- Plot 6: Correlation between -log_prob and entropy ---
    fig, ax = plt.subplots(figsize=(8, 8))
    # Subsample if too many points
    max_points = 10000
    if len(all_neg_lps) > max_points:
        idx = np.random.choice(len(all_neg_lps), max_points, replace=False)
        plot_nlp = all_neg_lps[idx]
        plot_ent = all_entropies[idx]
    else:
        plot_nlp = all_neg_lps
        plot_ent = all_entropies

    ax.scatter(plot_nlp, plot_ent, alpha=0.15, s=3, color="purple")
    ax.set_xlabel("-log_prob")
    ax.set_ylabel("Entropy")
    ax.set_title("Correlation: -log_prob vs Entropy")
    corr = np.corrcoef(all_neg_lps, all_entropies)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_nlp_entropy.png"), dpi=150)
    plt.close()

    print(f"  Saved 6 statistical plots to {output_dir}/")

    # --- Save summary stats to JSON ---
    summary = {
        "num_responses": len(results),
        "total_tokens": len(all_neg_lps),
        "neg_log_prob": {
            "mean": float(np.mean(all_neg_lps)),
            "std": float(np.std(all_neg_lps)),
            "median": float(np.median(all_neg_lps)),
            "min": float(np.min(all_neg_lps)),
            "max": float(np.max(all_neg_lps)),
            "p5": float(np.percentile(all_neg_lps, 5)),
            "p25": float(np.percentile(all_neg_lps, 25)),
            "p75": float(np.percentile(all_neg_lps, 75)),
            "p95": float(np.percentile(all_neg_lps, 95)),
        },
        "entropy": {
            "mean": float(np.mean(all_entropies)),
            "std": float(np.std(all_entropies)),
            "median": float(np.median(all_entropies)),
            "min": float(np.min(all_entropies)),
            "max": float(np.max(all_entropies)),
            "p5": float(np.percentile(all_entropies, 5)),
            "p25": float(np.percentile(all_entropies, 25)),
            "p75": float(np.percentile(all_entropies, 75)),
            "p95": float(np.percentile(all_entropies, 95)),
        },
        "correlation_nlp_entropy": float(corr),
    }
    with open(os.path.join(output_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary_stats.json")


# ---------------------------------------------------------------------------
# EDA: Correct vs Incorrect Analysis
# ---------------------------------------------------------------------------


def _build_eda_dataframe(results: list) -> pd.DataFrame:
    """Build a per-response DataFrame with summary features and correctness."""
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
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
        })
    return pd.DataFrame(rows)


def _print_group_stats(df: pd.DataFrame, label: str):
    """Print mean, std, covariance, correlation for a group."""
    features = ["neg_log_prob_mean", "entropy_mean", "response_length"]
    print(f"\n--- {label} (n={len(df)}) ---")
    for feat in features:
        vals = df[feat]
        print(f"  {feat:25s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={vals.median():.4f}")
    if len(df) > 1:
        print(f"\n  Covariance matrix:")
        print(df[features].cov().to_string(float_format=lambda x: f"{x:.6f}"))
        print(f"\n  Correlation matrix:")
        print(df[features].corr().to_string(float_format=lambda x: f"{x:.4f}"))


def generate_eda(results: list, output_dir: str):
    """EDA: compare correct vs incorrect responses with stats and plots.

    Analyses all data, correct subset, and incorrect subset. For each group
    computes mean, std, covariance matrix of (neg_log_prob_mean, entropy_mean,
    response_length). Generates comparison plots and runs Welch's t-test.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eda_dir = os.path.join(output_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)

    # Build DataFrame
    df = _build_eda_dataframe(results)
    if df.empty:
        print("  EDA: no data to analyze.")
        return

    has_correct = "correct" in df.columns and df["correct"].nunique() > 0
    df_correct = df[df["correct"]] if has_correct else pd.DataFrame()
    df_incorrect = df[~df["correct"]] if has_correct else pd.DataFrame()

    n_correct = len(df_correct)
    n_total = len(df)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # --- Print stats ---
    print(f"\n{'='*60}")
    print("EDA: CORRECT vs INCORRECT")
    print(f"{'='*60}")
    print(f"Accuracy: {n_correct}/{n_total} = {accuracy:.4f} ({accuracy*100:.1f}%)")

    _print_group_stats(df, "ALL")
    if len(df_correct) > 0:
        _print_group_stats(df_correct, "CORRECT")
    if len(df_incorrect) > 0:
        _print_group_stats(df_incorrect, "INCORRECT")

    # Welch's t-test
    if len(df_correct) >= 2 and len(df_incorrect) >= 2:
        try:
            from scipy import stats as sp_stats
            features = ["neg_log_prob_mean", "entropy_mean", "response_length"]
            print(f"\n--- Welch's t-test (correct vs incorrect) ---")
            for feat in features:
                t_stat, p_val = sp_stats.ttest_ind(
                    df_correct[feat], df_incorrect[feat], equal_var=False
                )
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {feat:25s}  t={t_stat:+.4f}  p={p_val:.6f} {sig}")
        except ImportError:
            print("  (scipy not installed, skipping t-test)")

    # --- Save CSV ---
    csv_path = os.path.join(eda_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # --- Plots ---
    features = ["neg_log_prob_mean", "entropy_mean", "response_length"]
    colors = {True: "#2ecc71", False: "#e74c3c"}
    labels = {True: "Correct", False: "Incorrect"}

    # Plot 1: Boxplots correct vs incorrect
    if len(df_correct) > 0 and len(df_incorrect) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, feat in zip(axes, features):
            data = [df_correct[feat].values, df_incorrect[feat].values]
            bp = ax.boxplot(data, labels=["Correct", "Incorrect"], patch_artist=True)
            bp["boxes"][0].set_facecolor(colors[True])
            bp["boxes"][1].set_facecolor(colors[False])
            ax.set_title(feat)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Correct vs Incorrect (accuracy={accuracy:.1%})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, "boxplots.png"), dpi=150)
        plt.close(fig)

    # Plot 2: Histograms correct vs incorrect
    if len(df_correct) > 0 and len(df_incorrect) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, feat in zip(axes, features):
            ax.hist(df_correct[feat], bins=30, alpha=0.6, color=colors[True],
                    label="Correct", density=True)
            ax.hist(df_incorrect[feat], bins=30, alpha=0.6, color=colors[False],
                    label="Incorrect", density=True)
            ax.set_title(feat)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Feature Distributions: Correct vs Incorrect", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, "histograms.png"), dpi=150)
        plt.close(fig)

    # Plot 3: Scatter neg_log_prob vs entropy (all data, colored by correctness)
    fig, ax = plt.subplots(figsize=(8, 6))
    for correct_val, group in df.groupby("correct"):
        ax.scatter(group["neg_log_prob_mean"], group["entropy_mean"],
                   alpha=0.5, s=20, color=colors.get(correct_val, "#999"),
                   label=labels.get(correct_val, str(correct_val)))
    ax.set_xlabel("neg_log_prob_mean")
    ax.set_ylabel("entropy_mean")
    ax.set_title("neg_log_prob vs Entropy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(eda_dir, "scatter_nlp_entropy.png"), dpi=150)
    plt.close(fig)

    # Plot 4: Scatter response_length vs neg_log_prob
    fig, ax = plt.subplots(figsize=(8, 6))
    for correct_val, group in df.groupby("correct"):
        ax.scatter(group["response_length"], group["neg_log_prob_mean"],
                   alpha=0.5, s=20, color=colors.get(correct_val, "#999"),
                   label=labels.get(correct_val, str(correct_val)))
    ax.set_xlabel("response_length (tokens)")
    ax.set_ylabel("neg_log_prob_mean")
    ax.set_title("Response Length vs neg_log_prob")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(eda_dir, "scatter_length_nlp.png"), dpi=150)
    plt.close(fig)

    # Plot 5: Scatter response_length vs entropy
    fig, ax = plt.subplots(figsize=(8, 6))
    for correct_val, group in df.groupby("correct"):
        ax.scatter(group["response_length"], group["entropy_mean"],
                   alpha=0.5, s=20, color=colors.get(correct_val, "#999"),
                   label=labels.get(correct_val, str(correct_val)))
    ax.set_xlabel("response_length (tokens)")
    ax.set_ylabel("entropy_mean")
    ax.set_title("Response Length vs Entropy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(eda_dir, "scatter_length_entropy.png"), dpi=150)
    plt.close(fig)

    # Save stats to JSON
    stats_out = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_total - n_correct,
        "n_total": n_total,
    }
    for label_key, sub_df in [("all", df), ("correct", df_correct), ("incorrect", df_incorrect)]:
        if len(sub_df) == 0:
            continue
        group_stats = {}
        for feat in features:
            group_stats[feat] = {
                "mean": float(sub_df[feat].mean()),
                "std": float(sub_df[feat].std()),
                "median": float(sub_df[feat].median()),
            }
        if len(sub_df) > 1:
            group_stats["covariance"] = sub_df[features].cov().to_dict()
            group_stats["correlation"] = sub_df[features].corr().to_dict()
        stats_out[label_key] = group_stats

    with open(os.path.join(eda_dir, "eda_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    print(f"  Saved EDA plots and stats to {eda_dir}/")


# ---------------------------------------------------------------------------
# HTML Visualization Helpers
# ---------------------------------------------------------------------------

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 20px;
        background: #fafafa;
        color: #333;
    }}
    h1 {{
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }}
    h2 {{
        color: #34495e;
        margin-top: 30px;
    }}
    .response-container {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .response-header {{
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #eee;
    }}
    .stats {{
        font-size: 0.85em;
        color: #7f8c8d;
        margin-bottom: 10px;
    }}
    .token-container {{
        display: flex;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 1px;
        padding-top: 10px;
        min-height: 80px;
    }}
    .line-break {{
        flex-basis: 100%;
        height: 0;
    }}
    .paragraph-break {{
        flex-basis: 100%;
        height: 12px;
    }}
    .token {{
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        cursor: pointer;
        min-width: 6px;
        position: relative;
    }}
    .token:hover {{
        outline: 1px solid #333;
        border-radius: 2px;
    }}
    .token .bar {{
        width: 100%;
        min-width: 6px;
        border-radius: 2px 2px 0 0;
    }}
    .token .annotation {{
        font-size: 0.5em;
        font-weight: bold;
        white-space: nowrap;
        pointer-events: none;
        margin-bottom: 1px;
    }}
    .token .token-text {{
        white-space: pre;
        font-size: 0.7em;
        max-width: 40px;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
        border-top: 1px solid #ddd;
        padding-top: 2px;
    }}
    .legend {{
        margin: 20px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }}
    .legend-bar {{
        height: 20px;
        border-radius: 4px;
        margin: 5px 0;
    }}
    .legend-labels {{
        display: flex;
        justify-content: space-between;
        font-size: 0.85em;
        color: #666;
    }}
    .prompt-text {{
        background: #e8f4f8;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-style: italic;
        color: #2c3e50;
        max-height: 150px;
        overflow-y: auto;
    }}
    .nav {{
        position: sticky;
        top: 0;
        background: #fff;
        padding: 10px 0;
        border-bottom: 1px solid #ddd;
        z-index: 100;
        margin-bottom: 20px;
    }}
    .nav a {{
        margin: 0 5px;
        text-decoration: none;
        color: #3498db;
    }}
    .summary-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }}
    .summary-table th, .summary-table td {{
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #eee;
    }}
    .summary-table th {{
        background: #f8f9fa;
        font-weight: 600;
    }}
</style>
</head>
<body>
"""

HTML_FOOTER = """
</body>
</html>
"""


def value_to_color(value: float, vmin: float, vmax: float, metric: str = "neg_log_prob") -> str:
    """Map a value to a color string. Low=green, High=red."""
    if vmax <= vmin:
        norm = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
    norm = max(0.0, min(1.0, norm))

    # Green (low) -> Yellow (mid) -> Red (high)
    if norm < 0.5:
        r = int(255 * (norm * 2))
        g = 180
        b = 50
    else:
        r = 255
        g = int(180 * (1 - (norm - 0.5) * 2))
        b = 50

    return f"rgb({r},{g},{b})"


def render_tokens_html(tokens: list, metric_key: str, vmin: float, vmax: float) -> str:
    """Render tokens as colored bars with height proportional to value."""
    max_bar_height = 80  # px
    parts = []
    for t in tokens:
        value = t[metric_key]
        color = value_to_color(value, vmin, vmax, metric_key)
        raw_token = t["token"]
        token_text = html_lib.escape(raw_token).replace("\n", "&#10;↵")
        annotation = f"{value:.2f}"

        # Compute bar height proportional to value
        if vmax <= vmin:
            norm = 0.5
        else:
            norm = (value - vmin) / (vmax - vmin)
        norm = max(0.05, min(1.0, norm))  # min 5% height so bar is always visible
        bar_height = int(norm * max_bar_height)

        parts.append(
            f'<span class="token" title="{metric_key}={value:.4f}, pos={t["position"]}">'
            f'<span class="annotation" style="color:{color}">{annotation}</span>'
            f'<span class="bar" style="height:{bar_height}px;background:{color}"></span>'
            f'<span class="token-text">{token_text}</span>'
            f'</span>'
        )
        # Insert line/paragraph breaks for newline tokens
        if "\n\n" in raw_token:
            parts.append('<div class="paragraph-break"></div>')
        elif "\n" in raw_token:
            parts.append('<div class="line-break"></div>')
    return "".join(parts)


def get_prompt_text(prompt) -> str:
    """Extract text from prompt (chat format or string)."""
    if isinstance(prompt, list):
        texts = []
        for msg in prompt:
            if isinstance(msg, dict):
                texts.append(f"[{msg.get('role', '?')}] {msg.get('content', '')}")
            else:
                texts.append(str(msg))
        return "\n".join(texts)
    return str(prompt)


# ---------------------------------------------------------------------------
# HTML Generation: -log_prob
# ---------------------------------------------------------------------------


def generate_neg_log_prob_html(results: list, output_path: str, max_samples: int = 50):
    """Generate HTML with -log_prob annotation above each token."""
    # Sample to keep HTML lightweight
    import random
    if len(results) > max_samples:
        sampled = random.sample(results, max_samples)
    else:
        sampled = results

    # Compute global min/max for color scaling
    all_vals = [t["neg_log_prob"] for r in sampled for t in r["tokens"]]
    if not all_vals:
        print("  No tokens found, skipping HTML generation.")
        return
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(HTML_HEADER.format(title="Reasoning Analysis: -log_prob per Token"))
        f.write("<h1>Reasoning Analysis: -log_prob per Token</h1>\n")

        # Legend
        f.write('<div class="legend">\n')
        f.write(f'<p><strong>Color scale:</strong> -log P(token) from {vmin:.2f} (green/low) to {vmax:.2f} (red/high)</p>\n')
        gradient = "linear-gradient(to right, rgb(0,180,50), rgb(255,180,50), rgb(255,0,50))"
        f.write(f'<div class="legend-bar" style="background: {gradient}"></div>\n')
        f.write(f'<div class="legend-labels"><span>Low -log_prob (confident)</span><span>High -log_prob (uncertain)</span></div>\n')
        f.write('</div>\n')

        # Summary table
        f.write('<h2>Summary Statistics</h2>\n')
        f.write('<table class="summary-table">\n')
        f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
        f.write(f'<tr><td>Total Responses</td><td>{len(results)}</td></tr>\n')
        f.write(f'<tr><td>Sampled Responses</td><td>{len(sampled)}</td></tr>\n')
        f.write(f'<tr><td>Total Tokens (sampled)</td><td>{len(all_vals)}</td></tr>\n')
        f.write(f'<tr><td>Mean -log_prob</td><td>{np.mean(all_vals):.4f}</td></tr>\n')
        f.write(f'<tr><td>Std -log_prob</td><td>{np.std(all_vals):.4f}</td></tr>\n')
        f.write(f'<tr><td>Median -log_prob</td><td>{np.median(all_vals):.4f}</td></tr>\n')
        f.write('</table>\n')

        # Navigation
        f.write('<div class="nav"><strong>Jump to:</strong> ')
        for i in range(len(sampled)):
            f.write(f'<a href="#resp-{i}">#{i}</a> ')
        f.write('</div>\n')

        # Each response
        for i, r in enumerate(sampled):
            f.write(f'<div class="response-container" id="resp-{i}">\n')
            f.write(f'<div class="response-header">Response #{i} '
                    f'(prompt_idx={r["prompt_idx"]}, sample_idx={r["sample_idx"]}, '
                    f'{r["num_tokens"]} tokens)</div>\n')

            # Prompt
            if "prompt" in r:
                prompt_text = html_lib.escape(get_prompt_text(r["prompt"]))
                f.write(f'<div class="prompt-text">{prompt_text}</div>\n')

            # Stats
            if "summary" in r:
                s = r["summary"]
                f.write(f'<div class="stats">'
                        f'mean={s["neg_log_prob_mean"]:.3f}, '
                        f'std={s["neg_log_prob_std"]:.3f}, '
                        f'max={s["neg_log_prob_max"]:.3f}, '
                        f'min={s["neg_log_prob_min"]:.3f}'
                        f'</div>\n')

            # Tokens
            f.write('<div class="token-container">\n')
            f.write(render_tokens_html(r["tokens"], "neg_log_prob", vmin, vmax))
            f.write('\n</div>\n')
            f.write('</div>\n')

        f.write(HTML_FOOTER)

    print(f"  Saved -log_prob HTML ({len(sampled)}/{len(results)} samples): {output_path}")


# ---------------------------------------------------------------------------
# HTML Generation: Entropy
# ---------------------------------------------------------------------------


def generate_entropy_html(results: list, output_path: str, max_samples: int = 50):
    """Generate HTML with entropy annotation above each token."""
    # Sample to keep HTML lightweight
    import random
    if len(results) > max_samples:
        sampled = random.sample(results, max_samples)
    else:
        sampled = results

    all_vals = [t["entropy"] for r in sampled for t in r["tokens"]]
    if not all_vals:
        print("  No tokens found, skipping HTML generation.")
        return
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(HTML_HEADER.format(title="Reasoning Analysis: Entropy per Token"))
        f.write("<h1>Reasoning Analysis: Entropy per Token</h1>\n")

        # Legend
        f.write('<div class="legend">\n')
        f.write(f'<p><strong>Color scale:</strong> Entropy H from {vmin:.2f} (green/low) to {vmax:.2f} (red/high)</p>\n')
        gradient = "linear-gradient(to right, rgb(0,180,50), rgb(255,180,50), rgb(255,0,50))"
        f.write(f'<div class="legend-bar" style="background: {gradient}"></div>\n')
        f.write(f'<div class="legend-labels"><span>Low Entropy (certain)</span><span>High Entropy (uncertain)</span></div>\n')
        f.write('</div>\n')

        # Summary table
        f.write('<h2>Summary Statistics</h2>\n')
        f.write('<table class="summary-table">\n')
        f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
        f.write(f'<tr><td>Total Responses</td><td>{len(results)}</td></tr>\n')
        f.write(f'<tr><td>Sampled Responses</td><td>{len(sampled)}</td></tr>\n')
        f.write(f'<tr><td>Total Tokens (sampled)</td><td>{len(all_vals)}</td></tr>\n')
        f.write(f'<tr><td>Mean Entropy</td><td>{np.mean(all_vals):.4f}</td></tr>\n')
        f.write(f'<tr><td>Std Entropy</td><td>{np.std(all_vals):.4f}</td></tr>\n')
        f.write(f'<tr><td>Median Entropy</td><td>{np.median(all_vals):.4f}</td></tr>\n')
        f.write('</table>\n')

        # Navigation
        f.write('<div class="nav"><strong>Jump to:</strong> ')
        for i in range(len(sampled)):
            f.write(f'<a href="#resp-{i}">#{i}</a> ')
        f.write('</div>\n')

        # Each response
        for i, r in enumerate(sampled):
            f.write(f'<div class="response-container" id="resp-{i}">\n')
            f.write(f'<div class="response-header">Response #{i} '
                    f'(prompt_idx={r["prompt_idx"]}, sample_idx={r["sample_idx"]}, '
                    f'{r["num_tokens"]} tokens)</div>\n')

            if "prompt" in r:
                prompt_text = html_lib.escape(get_prompt_text(r["prompt"]))
                f.write(f'<div class="prompt-text">{prompt_text}</div>\n')

            if "summary" in r:
                s = r["summary"]
                f.write(f'<div class="stats">'
                        f'mean={s["entropy_mean"]:.3f}, '
                        f'std={s["entropy_std"]:.3f}, '
                        f'max={s["entropy_max"]:.3f}, '
                        f'min={s["entropy_min"]:.3f}'
                        f'</div>\n')

            f.write('<div class="token-container">\n')
            f.write(render_tokens_html(r["tokens"], "entropy", vmin, vmax))
            f.write('\n</div>\n')
            f.write('</div>\n')

        f.write(HTML_FOOTER)

    print(f"  Saved entropy HTML ({len(sampled)}/{len(results)} samples): {output_path}")


# ---------------------------------------------------------------------------
# Attention Heatmap Visualization
# ---------------------------------------------------------------------------


def _load_npz_samples(internals_dir: str, max_samples: int = 50) -> list:
    """Load .npz hidden-state files, randomly sampling up to max_samples."""
    import glob as glob_mod
    import random

    npz_files = sorted(glob_mod.glob(os.path.join(internals_dir, "response_*.npz")))
    if not npz_files:
        print(f"  No .npz files found in {internals_dir}")
        return []

    if len(npz_files) > max_samples:
        npz_files = random.sample(npz_files, max_samples)
        npz_files.sort()

    samples = []
    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        tokens = list(data["tokens"])
        hidden_states = data["hidden_states"]  # (n_layers+1, response_len, hidden_dim)

        samples.append({
            "path": path,
            "meta": meta,
            "tokens": tokens,
            "hidden_states": hidden_states,
        })
    return samples


def _segment_into_sentences(tokens: list, boundary: int) -> list:
    """Split token list into sentence-like segments by `. `, `\\n\\n`, `?`, `!`.

    Returns list of dicts: {"label": str, "start": int, "end": int, "section": str}
    where start/end are token indices (exclusive end).
    """
    import re

    # Build full text with token boundaries
    segments = []
    current_start = 0
    accumulated = ""

    for i, tok in enumerate(tokens):
        accumulated += tok
        # Check if this token ends a sentence
        is_boundary = False
        if "\n\n" in tok:
            is_boundary = True
        elif re.search(r'[.?!]\s*$', accumulated):
            is_boundary = True
        elif re.search(r'[.?!]$', tok):
            is_boundary = True

        if is_boundary and i > current_start:
            section = "thinking" if current_start < boundary else "output"
            if current_start < boundary <= i + 1:
                # Sentence spans boundary — split it
                if current_start < boundary:
                    segments.append({
                        "start": current_start,
                        "end": boundary,
                        "section": "thinking",
                    })
                if boundary < i + 1:
                    segments.append({
                        "start": boundary,
                        "end": i + 1,
                        "section": "output",
                    })
            else:
                segments.append({
                    "start": current_start,
                    "end": i + 1,
                    "section": section,
                })
            current_start = i + 1
            accumulated = ""

    # Remaining tokens
    if current_start < len(tokens):
        section = "thinking" if current_start < boundary else "output"
        segments.append({
            "start": current_start,
            "end": len(tokens),
            "section": section,
        })

    # Create short labels
    for idx, seg in enumerate(segments):
        seg_tokens = tokens[seg["start"]:seg["end"]]
        text = "".join(seg_tokens).strip()
        if len(text) > 30:
            text = text[:27] + "..."
        seg["label"] = f"S{idx}[{seg['section'][0].upper()}]: {text}"

    return segments


def _compute_sentence_attention(attn_matrix: np.ndarray, segments: list,
                                row_offset: int, col_offset: int) -> np.ndarray:
    """Compute sentence-level attention from token-level attention matrix.

    attn_matrix: (rows, cols) token-level attention (already head-averaged).
    segments: list of sentence segments with start/end indices.
    row_offset, col_offset: global token offsets for rows and cols.

    Returns: (n_segments_row, n_segments_col) matrix.
    """
    n = len(segments)
    sent_attn = np.zeros((n, n))

    for i, seg_i in enumerate(segments):
        ri_start = seg_i["start"] - row_offset
        ri_end = seg_i["end"] - row_offset
        # Clip to valid range
        ri_start = max(0, ri_start)
        ri_end = min(attn_matrix.shape[0], ri_end)
        if ri_start >= ri_end:
            continue

        for j, seg_j in enumerate(segments):
            cj_start = seg_j["start"] - col_offset
            cj_end = seg_j["end"] - col_offset
            cj_start = max(0, cj_start)
            cj_end = min(attn_matrix.shape[1], cj_end)
            if cj_start >= cj_end:
                continue

            block = attn_matrix[ri_start:ri_end, cj_start:cj_end]
            sent_attn[i, j] = block.mean()

    return sent_attn


def _reconstruct_attention_for_sample(model, sample: dict, layer_idx: int):
    """Reconstruct full-sequence attention for one sample at one layer.

    Uses hidden states stored in the .npz + model weights to reconstruct
    attention via the reconstruct module (RoPE, GQA, QK-Norm handled).

    Returns: (num_heads, response_len, response_len) attention matrix.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention_analysis.reconstruct import reconstruct_attention
    import torch

    meta = sample["meta"]
    prompt_len = meta["prompt_len"]
    hidden_states = sample["hidden_states"]  # (n_layers+1, response_len, hidden_dim)

    # hidden_states[layer_idx] = input to layer layer_idx
    # But we only saved response tokens (from gen_start onward),
    # so we need to account for prompt_len in position_ids.
    response_len = hidden_states.shape[1]

    # Build full hidden state: need to pass through model's layer
    hs = torch.from_numpy(hidden_states[layer_idx]).unsqueeze(0)  # (1, resp_len, hidden)
    hs = hs.to(dtype=torch.bfloat16, device=next(model.parameters()).device)

    # Position IDs: response tokens start at prompt_len
    position_ids = torch.arange(
        prompt_len, prompt_len + response_len,
        device=hs.device
    ).unsqueeze(0)

    with torch.no_grad():
        attn = reconstruct_attention(model, layer_idx, hs, position_ids)

    return attn.cpu().numpy()  # (num_heads, response_len, response_len)


def _normalize_attention_for_viz(matrix: np.ndarray) -> np.ndarray:
    """Normalize attention matrix to enhance visual contrast.

    Uses percentile-based normalization: maps [p2, p98] to [0, 1],
    then clips. This makes the color variation much more visible
    even when most values are near zero (typical for attention).
    """
    if matrix.size == 0:
        return matrix
    flat = matrix.flatten()
    # Remove exact zeros from percentile computation (causal mask creates many)
    nonzero = flat[flat > 0]
    if len(nonzero) == 0:
        return matrix
    vmin = np.percentile(nonzero, 2)
    vmax = np.percentile(nonzero, 98)
    if vmax <= vmin:
        vmax = vmin + 1e-8
    normed = (matrix - vmin) / (vmax - vmin)
    return np.clip(normed, 0.0, 1.0)


def _render_attention_heatmap_html(
    matrix: np.ndarray,
    row_tokens: list,
    col_tokens: list,
    title: str,
    output_path: str,
    cmap_name: str = "Blues",
):
    """Render an attention heatmap as a fully interactive HTML file.

    Every token is shown on both axes. Cells are colored by attention weight
    with enhanced contrast (percentile normalization). Supports zoom/scroll
    for large matrices. Hover shows exact value.
    """
    n_rows, n_cols = matrix.shape

    # Normalize for color mapping
    normed = _normalize_attention_for_viz(matrix)

    # Generate color for each cell
    # Blues-like: 0 → white (255,255,255), 1 → dark blue (8,48,107)
    if "Orange" in cmap_name:
        def val_to_rgb(v):
            r = 255
            g = int(255 - v * 190)
            b = int(255 - v * 225)
            return f"rgb({r},{g},{b})"
    else:
        def val_to_rgb(v):
            r = int(255 - v * 247)
            g = int(255 - v * 207)
            b = int(255 - v * 148)
            return f"rgb({r},{g},{b})"

    # Compute cell size: at least 20px, scale with token count
    cell_size = max(20, min(40, 800 // max(n_rows, n_cols, 1)))

    # Escape token strings for HTML
    def esc(s):
        return html_lib.escape(str(s)).replace(" ", "&nbsp;")

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html_lib.escape(title)}</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #fafafa; }}
h1 {{ font-size: 18px; color: #2c3e50; }}
.info {{ font-size: 13px; color: #555; margin-bottom: 10px; }}
.heatmap-wrap {{
    overflow: auto;
    max-width: 95vw;
    max-height: 90vh;
    border: 1px solid #ccc;
    position: relative;
}}
table {{
    border-collapse: collapse;
    font-size: 10px;
}}
td, th {{
    width: {cell_size}px;
    min-width: {cell_size}px;
    max-width: {cell_size}px;
    height: {cell_size}px;
    text-align: center;
    padding: 0;
    border: 1px solid rgba(200,200,200,0.3);
    overflow: hidden;
}}
th {{
    background: #f8f8f8;
    position: sticky;
    z-index: 2;
    font-weight: normal;
    font-size: 9px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: {cell_size}px;
    padding: 2px;
}}
th.col-header {{
    top: 0;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    height: 80px;
    max-height: 80px;
    vertical-align: bottom;
}}
th.row-header {{
    left: 0;
    text-align: right;
    padding-right: 4px;
    max-width: 120px;
    width: 120px;
    min-width: 120px;
}}
th.corner {{
    top: 0; left: 0; z-index: 3;
    background: #f0f0f0;
    width: 120px; min-width: 120px;
    height: 80px;
}}
td:hover {{
    outline: 2px solid #333;
    z-index: 1;
}}
.tooltip {{
    display: none;
    position: fixed;
    background: rgba(0,0,0,0.85);
    color: #fff;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 999;
    white-space: pre;
}}
.legend {{
    margin: 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}}
.legend-bar {{
    width: 200px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #ccc;
}}
</style>
</head>
<body>
<h1>{html_lib.escape(title)}</h1>
<div class="info">
  Matrix size: {n_rows} (query) &times; {n_cols} (key) &nbsp;|&nbsp;
  Raw value range: [{matrix.min():.6f}, {matrix.max():.6f}] &nbsp;|&nbsp;
  Non-zero cells: {np.count_nonzero(matrix)}/{matrix.size}
</div>
""")

    # Legend
    if "Orange" in cmap_name:
        grad = "linear-gradient(to right, rgb(255,255,255), rgb(255,65,30))"
    else:
        grad = "linear-gradient(to right, rgb(255,255,255), rgb(8,48,107))"
    html_parts.append(f"""<div class="legend">
  <span>Low attention</span>
  <div class="legend-bar" style="background: {grad}"></div>
  <span>High attention</span>
  <span style="color:#888">(percentile-normalized for contrast)</span>
</div>
""")

    html_parts.append('<div class="heatmap-wrap">\n<table>\n')

    # Header row with column tokens
    html_parts.append('<tr><th class="corner">Query \\ Key</th>')
    for j, tok in enumerate(col_tokens):
        html_parts.append(f'<th class="col-header" title="col {j}: {esc(tok)}">{esc(tok)}</th>')
    html_parts.append('</tr>\n')

    # Data rows
    for i in range(n_rows):
        html_parts.append(f'<tr><th class="row-header" title="row {i}: {esc(row_tokens[i])}">{esc(row_tokens[i])}</th>')
        for j in range(n_cols):
            raw_val = matrix[i, j]
            norm_val = normed[i, j]
            color = val_to_rgb(float(norm_val))
            html_parts.append(
                f'<td style="background:{color}" '
                f'data-r="{i}" data-c="{j}" data-v="{raw_val:.6f}">'
                f'</td>'
            )
        html_parts.append('</tr>\n')

    html_parts.append('</table>\n</div>\n')

    # Tooltip + hover script
    html_parts.append("""
<div class="tooltip" id="tooltip"></div>
<script>
const tip = document.getElementById('tooltip');
document.querySelector('.heatmap-wrap').addEventListener('mousemove', function(e) {
    const td = e.target.closest('td[data-v]');
    if (td) {
        const r = td.dataset.r, c = td.dataset.c, v = td.dataset.v;
        const rt = document.querySelectorAll('th.row-header')[r];
        const ct = document.querySelectorAll('th.col-header')[c];
        const rn = rt ? rt.textContent : r;
        const cn = ct ? ct.textContent : c;
        tip.innerHTML = `Query[${r}]: ${rn}\\nKey[${c}]: ${cn}\\nAttention: ${v}`;
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY + 14) + 'px';
    } else {
        tip.style.display = 'none';
    }
});
document.querySelector('.heatmap-wrap').addEventListener('mouseleave', function() {
    tip.style.display = 'none';
});
</script>
</body></html>""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))


def generate_attention_heatmaps(model_path: str, internals_dir: str,
                                output_dir: str, max_samples: int = 50,
                                attn_impl: str = "eager",
                                layers: list[int] | None = None,
                                no_png: bool = False):
    """Generate attention heatmaps as interactive HTML files + optional PNG.

    Reconstructs attention matrices from hidden states + model weights.
    All tokens are shown on axes (no truncation). Color is percentile-
    normalized to ensure visible contrast even with sparse attention.

    Args:
        layers: List of layer indices (0-based) to visualize.
                None = last layer only.
                Each layer gets its own subdirectory: attention_heatmaps/layer_XX/
        no_png: If True, skip PNG generation (saves significant RAM for
                large token counts where matplotlib figures become huge).
    """
    if not no_png:
        import matplotlib
        matplotlib.use("Agg")
        _configure_matplotlib()
        import matplotlib.pyplot as plt
    import torch
    from transformers import AutoModelForCausalLM

    samples = _load_npz_samples(internals_dir, max_samples)
    if not samples:
        return

    # Load model (only weights needed, no inference)
    print(f"  Loading model for attention reconstruction (attn={attn_impl}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers

    # Resolve layer list
    if layers is not None and len(layers) > 0:
        viz_layers = []
        for l in layers:
            if l < 0 or l >= n_layers:
                print(f"  WARNING: layer {l} out of range [0, {n_layers-1}], skipping")
            else:
                viz_layers.append(l)
        if not viz_layers:
            print(f"  No valid layers specified, using last layer")
            viz_layers = [n_layers - 1]
    else:
        viz_layers = [n_layers - 1]

    print(f"  Visualizing {len(viz_layers)} layer(s): {viz_layers} (total: {n_layers})")

    for viz_layer in viz_layers:
        # Each layer gets its own subdirectory
        if len(viz_layers) == 1:
            attn_dir = os.path.join(output_dir, "attention_heatmaps")
        else:
            attn_dir = os.path.join(output_dir, "attention_heatmaps", f"layer_{viz_layer:02d}")
        os.makedirs(attn_dir, exist_ok=True)

        print(f"\n  --- Layer {viz_layer}/{n_layers-1} → {attn_dir}/ ---")

        for sample_idx, sample in enumerate(samples):
            meta = sample["meta"]
            tokens = sample["tokens"]
            think_boundary = meta["think_boundary"]
            think_len = meta["think_len"]
            out_len = meta["out_len"]

            if think_len == 0 and out_len == 0:
                continue

            # Reconstruct attention from hidden states
            # (num_heads, response_len, response_len)
            try:
                attn_full = _reconstruct_attention_for_sample(model, sample, viz_layer)
            except Exception as e:
                print(f"  WARNING: Failed to reconstruct sample {sample_idx} layer {viz_layer}: {e}")
                continue

            # Average across heads → (response_len, response_len)
            attn_avg = attn_full.mean(axis=0).astype(np.float32)

            # Debug: report value stats for diagnostics
            nonzero_vals = attn_avg[attn_avg > 0]
            if len(nonzero_vals) > 0:
                print(f"    sample {sample_idx} L{viz_layer}: nonzero={len(nonzero_vals)}/{attn_avg.size}, "
                      f"range=[{nonzero_vals.min():.6f}, {nonzero_vals.max():.6f}], "
                      f"median={np.median(nonzero_vals):.6f}")
            else:
                print(f"    sample {sample_idx} L{viz_layer}: WARNING attn all zeros!")

            layer_tag = f"L{viz_layer}"

            # --- Token-level heatmaps (HTML + PNG) ---

            # Think→Think sub-matrix
            if think_boundary is not None and think_len > 0:
                tt = attn_avg[:think_boundary, :think_boundary]
                think_tokens = tokens[:think_boundary]
                title = (f"Think→Think Attention (sample {sample_idx}, "
                         f"layer {viz_layer}, head avg, {len(think_tokens)} tokens)")

                # HTML (primary — always full detail)
                _render_attention_heatmap_html(
                    tt, think_tokens, think_tokens, title,
                    os.path.join(attn_dir, f"sample_{sample_idx:03d}_think_think.html"),
                    cmap_name="Blues",
                )

                # PNG (with percentile normalization + all tokens shown)
                if not no_png:
                    normed_tt = _normalize_attention_for_viz(tt)
                    fig_w = max(10, len(think_tokens) * 0.35)
                    fig_h = max(8, len(think_tokens) * 0.30)
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    im = ax.imshow(normed_tt, aspect="auto", cmap="Blues", vmin=0, vmax=1)
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel("Key (thinking tokens)")
                    ax.set_ylabel("Query (thinking tokens)")
                    ax.set_xticks(range(len(think_tokens)))
                    ax.set_xticklabels(think_tokens, rotation=90, fontsize=max(4, min(7, 500 // len(think_tokens))))
                    ax.set_yticks(range(len(think_tokens)))
                    ax.set_yticklabels(think_tokens, fontsize=max(4, min(7, 500 // len(think_tokens))))
                    plt.colorbar(im, ax=ax, shrink=0.8, label="attention (normalized)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(attn_dir, f"sample_{sample_idx:03d}_think_think_tokens.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close()

            # Output→Think sub-matrix
            if think_boundary is not None and think_len > 0 and out_len > 0:
                ot = attn_avg[think_boundary:, :think_boundary]
                think_tokens = tokens[:think_boundary]
                out_tokens = tokens[think_boundary:]
                title = (f"Output→Think Attention (sample {sample_idx}, "
                         f"layer {viz_layer}, head avg, {len(out_tokens)}x{len(think_tokens)})")

                # HTML
                _render_attention_heatmap_html(
                    ot, out_tokens, think_tokens, title,
                    os.path.join(attn_dir, f"sample_{sample_idx:03d}_out_think.html"),
                    cmap_name="Oranges",
                )

                # PNG
                if not no_png:
                    normed_ot = _normalize_attention_for_viz(ot)
                    fig_w = max(10, len(think_tokens) * 0.35)
                    fig_h = max(6, len(out_tokens) * 0.35)
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    im = ax.imshow(normed_ot, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel("Key (thinking tokens)")
                    ax.set_ylabel("Query (output tokens)")
                    ax.set_xticks(range(len(think_tokens)))
                    ax.set_xticklabels(think_tokens, rotation=90, fontsize=max(4, min(7, 500 // len(think_tokens))))
                    ax.set_yticks(range(len(out_tokens)))
                    ax.set_yticklabels(out_tokens, fontsize=max(4, min(7, 500 // len(out_tokens))))
                    plt.colorbar(im, ax=ax, shrink=0.8, label="attention (normalized)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(attn_dir, f"sample_{sample_idx:03d}_out_think_tokens.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close()

            # --- Sentence-level heatmaps ---
            boundary = think_boundary if think_boundary is not None else 0
            segments = _segment_into_sentences(tokens, boundary)
            if len(segments) < 2:
                continue

            labels = [s["label"] for s in segments]

            # Sentence-level from full response attention
            sent_attn = _compute_sentence_attention(attn_avg, segments,
                                                    row_offset=0, col_offset=0)

            # HTML
            _render_attention_heatmap_html(
                sent_attn, labels, labels,
                f"Sentence-level Attention (sample {sample_idx}, layer {viz_layer})",
                os.path.join(attn_dir, f"sample_{sample_idx:03d}_sentences.html"),
                cmap_name="Blues",
            )

            # PNG
            if not no_png:
                normed_sent = _normalize_attention_for_viz(sent_attn)
                fig_size = max(8, len(segments) * 0.6)
                fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                im = ax.imshow(normed_sent, aspect="auto", cmap="Blues", vmin=0, vmax=1)
                ax.set_title(f"Sentence-level Attention (sample {sample_idx}, layer {viz_layer})")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=7)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=7)
                plt.colorbar(im, ax=ax, shrink=0.8, label="attention (normalized)")
                plt.tight_layout()
                plt.savefig(os.path.join(attn_dir, f"sample_{sample_idx:03d}_sentences.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()

            if (sample_idx + 1) % 10 == 0:
                print(f"    [{sample_idx+1}/{len(samples)}] heatmaps generated")

        print(f"  Layer {viz_layer}: done ({len(samples)} samples)")

    # Free model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Saved attention heatmaps ({len(samples)} samples x {len(viz_layers)} layers)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize reasoning analysis results")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to analysis results (.json or .jsonl)")
    parser.add_argument("--output_dir", type=str, default="reasoning_analysis/outputs/visualizations",
                        help="Output directory for plots and HTML files")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating statistical plots (useful if matplotlib not available)")
    parser.add_argument("--internals_dir", type=str, default=None,
                        help="Directory containing .npz files with hidden states "
                             "(e.g. reasoning_analysis/outputs/internals)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path for attention reconstruction from hidden states. "
                             "Required if --internals_dir is provided.")
    parser.add_argument("--attn_impl", type=str, default="eager",
                        choices=["flash_attention_2", "eager"],
                        help="Attention implementation for model loading during "
                             "reconstruction (default: eager)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices (0-based) for attention heatmaps. "
                             "Accepts one or more values, e.g. --layers 0 12 27. "
                             "Default: last layer only.")
    parser.add_argument("--no_png", action="store_true",
                        help="Skip PNG heatmap generation to save RAM. "
                             "Only HTML heatmaps will be produced.")
    args = parser.parse_args()

    print(f"Loading results from {args.input_path} ...")
    results = load_results(args.input_path)
    print(f"Loaded {len(results)} responses")

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate statistical plots
    if not args.no_plots:
        print("Generating statistical plots ...")
        try:
            generate_statistics_plots(results, args.output_dir)
        except ImportError:
            print("  WARNING: matplotlib not available. Skipping plots.")
            print("  Install with: pip install matplotlib")

    # EDA: correct vs incorrect analysis
    has_correct = any("correct" in r for r in results)
    if has_correct and not args.no_plots:
        print("Generating EDA (correct vs incorrect) ...")
        try:
            generate_eda(results, args.output_dir)
        except ImportError as e:
            print(f"  WARNING: EDA skipped ({e})")
    elif not has_correct:
        print("Skipping EDA: 'correct' field not found. Re-run evaluate.py to add accuracy.")

    # Generate HTML files
    print("Generating -log_prob HTML ...")
    generate_neg_log_prob_html(results, os.path.join(args.output_dir, "neg_log_prob.html"))

    print("Generating entropy HTML ...")
    generate_entropy_html(results, os.path.join(args.output_dir, "entropy.html"))

    # Generate attention heatmaps if internals_dir provided
    if args.internals_dir:
        if not args.model_path:
            print("WARNING: --model_path required for attention heatmaps. Skipping.")
        else:
            print("Generating attention heatmaps (reconstructing from hidden states) ...")
            try:
                generate_attention_heatmaps(
                    model_path=args.model_path,
                    internals_dir=args.internals_dir,
                    output_dir=args.output_dir,
                    attn_impl=args.attn_impl,
                    layers=args.layers,
                    no_png=args.no_png,
                )
            except ImportError as e:
                print(f"  WARNING: Attention heatmaps skipped ({e})")

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Files generated:")
    for root, dirs, files in os.walk(args.output_dir):
        rel = os.path.relpath(root, args.output_dir)
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size_kb = os.path.getsize(fpath) / 1024
            display = fname if rel == "." else os.path.join(rel, fname)
            print(f"  {display} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
