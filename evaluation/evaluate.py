"""
Evaluation script for ADPO-trained models.

Generates responses for eval datasets and computes accuracy metrics.
Uses vLLM for fast batched inference.

Usage:
    python evaluation/evaluate.py \
        --model_path checkpoints/adpo-qwen-7b \
        --datasets math500 gsm8k_test aime_2024 \
        --output_dir results/ \
        --n_samples 1 \
        --temperature 0.0
"""

import argparse
import json
import math
import os
from collections import Counter

import pandas as pd
import torch


def _parse_field(val):
    """Parse a field that may be a JSON string, dict, list, or ndarray."""
    if isinstance(val, str):
        return json.loads(val)
    if isinstance(val, (dict, list)):
        return val
    # numpy ndarray or other — convert to list
    if hasattr(val, 'tolist'):
        return val.tolist()
    return val


def load_eval_data(dataset_name, data_dir="data/processed/eval"):
    path = os.path.join(data_dir, f"{dataset_name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Eval data not found: {path}. "
            f"Run: python data/prepare_datasets.py --dataset {dataset_name}"
        )
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        prompt = _parse_field(row["prompt"])
        reward_model = _parse_field(row["reward_model"])
        extra_info = _parse_field(row.get("extra_info", "{}"))
        gt = reward_model["ground_truth"] if isinstance(reward_model, dict) else str(reward_model)
        records.append({
            "data_source": row["data_source"],
            "prompt": prompt,
            "ground_truth": gt,
            "extra_info": extra_info if isinstance(extra_info, dict) else {},
        })
    return records


def generate_responses_vllm(
    model_path, prompts, n_samples=1, temperature=0.0,
    max_tokens=2048, top_p=0.95, tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    formatted_prompts = []
    for p in prompts:
        text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(formatted_prompts, sampling_params)
    results = []
    for output in outputs:
        samples = [o.text for o in output.outputs]
        results.append(samples)
    return results


def evaluate_dataset(
    model_path, dataset_name, data_dir="data/processed/eval",
    n_samples=1, temperature=0.0, max_tokens=2048, tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
):
    from adpo.reward_functions import compute_score

    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}")

    records = load_eval_data(dataset_name, data_dir)
    print(f"Loaded {len(records)} problems")

    prompts = [r["prompt"] for r in records]
    responses = generate_responses_vllm(
        model_path=model_path, prompts=prompts, n_samples=n_samples,
        temperature=temperature, max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    results = []
    total = 0

    for record, response_list in zip(records, responses):
        scores = []
        for response in response_list:
            score = compute_score(
                data_source=record["data_source"],
                solution_str=response,
                ground_truth=record["ground_truth"],
                extra_info=record["extra_info"],
            )
            scores.append(score)

        n_correct = sum(1 for s in scores if s > 0.5)
        results.append({
            "data_source": record["data_source"],
            "ground_truth": record["ground_truth"],
            "responses": response_list,
            "scores": scores,
            "n_correct": n_correct,
        })
        total += 1

    # --- Metrics ---
    n = n_samples
    metrics = {
        "dataset": dataset_name,
        "num_problems": total,
        "n_samples": n,
        "temperature": temperature,
    }

    # Pass@1: fraction of problems where at least 1 sample is correct
    # When n=1, this is standard greedy accuracy
    pass_at_1_count = sum(1 for r in results if r["n_correct"] > 0)
    metrics["pass@1"] = pass_at_1_count / total if total > 0 else 0.0

    # Avg@n: average score across all samples per problem, then mean across problems
    avg_scores = [sum(r["scores"]) / len(r["scores"]) for r in results]
    metrics["avg@n"] = sum(avg_scores) / total if total > 0 else 0.0

    if n > 1:
        # Pass@k (unbiased estimator): E[1 - C(n-c, k) / C(n, k)]
        # where c = number of correct samples out of n
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
                    # 1 - C(n-c, k) / C(n, k)
                    pass_at_k_sum += 1.0 - math.comb(n - c, k) / math.comb(n, k)
            metrics[f"pass@{k}"] = pass_at_k_sum / total if total > 0 else 0.0

        # Maj@n: majority voting — most common answer wins
        maj_correct = 0
        for r in results:
            # Use score > 0.5 as "correct", count majority
            correct_count = r["n_correct"]
            if correct_count > n / 2:
                maj_correct += 1
        metrics[f"maj@{n}"] = maj_correct / total if total > 0 else 0.0

    # Print results
    print(f"  Pass@1:  {metrics['pass@1']:.4f} ({pass_at_1_count}/{total})")
    print(f"  Avg@{n}:  {metrics['avg@n']:.4f}")
    if n > 1:
        for k in [1, 2, 4, 8, 16, 32, 64]:
            if f"pass@{k}" not in metrics:
                break
            print(f"  Pass@{k}: {metrics[f'pass@{k}']:.4f}")
        print(f"  Maj@{n}:  {metrics[f'maj@{n}']:.4f}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="ADPO Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["math500", "gsm8k_test"])
    parser.add_argument("--data_dir", type=str, default="data/processed/eval")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = []

    for dataset_name in args.datasets:
        try:
            metrics, results = evaluate_dataset(
                model_path=args.model_path, dataset_name=dataset_name,
                data_dir=args.data_dir, n_samples=args.n_samples,
                temperature=args.temperature, max_tokens=args.max_tokens,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            all_metrics.append(metrics)
            result_path = os.path.join(args.output_dir, f"{dataset_name}_results.json")
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [ERROR] {dataset_name}: {e}")
            all_metrics.append({"dataset": dataset_name, "error": str(e)})

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Dataset':<20} {'Pass@1':>8} {'Avg@n':>8}"
    if args.n_samples > 1:
        header += f" {'Maj@n':>8}"
    header += f" {'Total':>8}"
    print(header)
    print("-" * 70)
    for m in all_metrics:
        if "error" in m:
            print(f"{m['dataset']:<20} {'ERROR':>8}")
        else:
            n = m['n_samples']
            line = f"{m['dataset']:<20} {m['pass@1']:>8.4f} {m['avg@n']:>8.4f}"
            if n > 1:
                line += f" {m.get(f'maj@{n}', 0):>8.4f}"
            line += f" {m['num_problems']:>8}"
            print(line)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
