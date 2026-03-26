#!/usr/bin/env python3
"""Re-evaluate results JSON with updated scoring functions.

Reads a results JSON file (produced by evaluate.py), re-scores all responses
using the latest compute_score, and prints updated metrics.

Usage:
    PYTHONPATH=. python evaluation/evaluate_json.py results/global_step_250/aime_2025_results.json
    PYTHONPATH=. python evaluation/evaluate_json.py results/*.json --save
"""

import argparse
import json
import math
import os
import sys


def rescore_results(results, compute_score_fn):
    """Re-score all responses in a results list."""
    rescored = []
    for r in results:
        scores = []
        for response in r["responses"]:
            score = compute_score_fn(
                data_source=r["data_source"],
                solution_str=response,
                ground_truth=r["ground_truth"],
                extra_info=r.get("extra_info", {}),
            )
            scores.append(score)

        n_correct = sum(1 for s in scores if s > 0.5)
        rescored.append({
            **r,
            "scores": scores,
            "n_correct": n_correct,
            "old_scores": r.get("scores", []),
        })
    return rescored


def compute_metrics(results, n_samples):
    """Compute Pass@k, Avg@n, Maj@n metrics."""
    total = len(results)
    if total == 0:
        return {}

    n = n_samples
    metrics = {"num_problems": total, "n_samples": n}

    # Pass@1
    pass_at_1_count = sum(1 for r in results if r["n_correct"] > 0)
    metrics["pass@1"] = pass_at_1_count / total

    # Avg@n
    avg_scores = [sum(r["scores"]) / len(r["scores"]) for r in results]
    metrics["avg@n"] = sum(avg_scores) / total

    if n > 1:
        # Pass@k (unbiased estimator)
        for k in [1, 2, 4, 8, 16, 32, 64]:
            if k > n:
                break
            pass_at_k_sum = 0.0
            for r in results:
                c = r["n_correct"]
                if c >= k:
                    pass_at_k_sum += 1.0
                elif n - c < k:
                    pass_at_k_sum += 1.0
                else:
                    pass_at_k_sum += 1.0 - math.comb(n - c, k) / math.comb(n, k)
            metrics[f"pass@{k}"] = pass_at_k_sum / total

        # Maj@n
        maj_correct = sum(1 for r in results if r["n_correct"] > n / 2)
        metrics[f"maj@{n}"] = maj_correct / total

    return metrics


def print_metrics(metrics, n):
    """Pretty-print metrics."""
    total = metrics["num_problems"]
    pass1 = metrics["pass@1"]
    print(f"  Pass@1:  {pass1:.4f} ({int(pass1 * total)}/{total})")
    print(f"  Avg@{n}:  {metrics['avg@n']:.4f}")
    if n > 1:
        for k in [1, 2, 4, 8, 16, 32, 64]:
            key = f"pass@{k}"
            if key not in metrics:
                break
            print(f"  Pass@{k}: {metrics[key]:.4f}")
        print(f"  Maj@{n}:  {metrics.get(f'maj@{n}', 0):.4f}")


def print_diff(old_results, new_results):
    """Show problems where scores changed."""
    changed = 0
    for old, new in zip(old_results, new_results):
        old_scores = old.get("scores", [])
        new_scores = new["scores"]
        if old_scores != new_scores:
            changed += 1
            old_c = sum(1 for s in old_scores if s > 0.5)
            new_c = new["n_correct"]
            gt = new["ground_truth"]
            ds = new["data_source"]
            if changed <= 10:
                # Show first response's answer for context
                resp0 = new["responses"][0][:100].replace("\n", "\\n") if new["responses"] else ""
                print(f"  [{ds}] gt={gt!r} | old={old_c}/{len(old_scores)} → "
                      f"new={new_c}/{len(new_scores)} | \"{resp0}...\"")
    if changed > 10:
        print(f"  ... and {changed - 10} more")
    print(f"  Total changed: {changed}/{len(new_results)}")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate results JSON with updated scoring")
    parser.add_argument("files", nargs="+", help="Path(s) to *_results.json files")
    parser.add_argument("--save", action="store_true", help="Overwrite JSON files with re-scored results")
    parser.add_argument("--output_dir", default=None, help="Save re-scored results to this directory instead of overwriting")
    args = parser.parse_args()

    from adpo.reward_functions import compute_score

    for filepath in args.files:
        filename = os.path.basename(filepath)
        print(f"\n{'='*60}")
        print(f"Re-scoring: {filepath}")
        print(f"{'='*60}")

        with open(filepath) as f:
            old_results = json.load(f)

        if not old_results:
            print("  (empty)")
            continue

        n_samples = len(old_results[0].get("responses", []))
        if n_samples == 0:
            print("  (no responses)")
            continue

        # Old metrics
        print(f"\n  --- OLD scores ({len(old_results)} problems, n={n_samples}) ---")
        old_for_metrics = []
        for r in old_results:
            old_for_metrics.append({
                **r,
                "n_correct": sum(1 for s in r.get("scores", []) if s > 0.5),
            })
        old_metrics = compute_metrics(old_for_metrics, n_samples)
        print_metrics(old_metrics, n_samples)

        # Re-score
        new_results = rescore_results(old_results, compute_score)

        print(f"\n  --- NEW scores ---")
        new_metrics = compute_metrics(new_results, n_samples)
        print_metrics(new_metrics, n_samples)

        # Diff
        print(f"\n  --- Changes ---")
        print_diff(old_results, new_results)

        # Save
        if args.save or args.output_dir:
            # Remove old_scores before saving
            save_results = []
            for r in new_results:
                save_r = {k: v for k, v in r.items() if k != "old_scores"}
                save_results.append(save_r)

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, filename)
            else:
                out_path = filepath

            with open(out_path, "w") as f:
                json.dump(save_results, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
