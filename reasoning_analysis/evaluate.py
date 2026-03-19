"""
Reasoning Analysis: Evaluate models and compute per-token -log_prob and entropy.

For each generated token, computes:
  - neg_log_prob: -log P(token | context)
  - entropy: H = -sum_v P(v|context) * log P(v|context)  (over full vocabulary)

Saves results to a JSONL file where each line is one (prompt, response) pair
with token-level statistics.

Usage:
    python reasoning_analysis/evaluate.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --dataset_path data/processed/eval/math500.parquet \
        --output_path reasoning_analysis/outputs/analysis.jsonl \
        --max_samples 50 \
        --temperature 0.6 \
        --max_tokens 2048 \
        --tensor_parallel_size 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


def load_dataset_from_parquet(parquet_path: str, max_samples: int = -1):
    """Load evaluation data from parquet file."""
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        record = {
            "data_source": row.get("data_source", "unknown"),
            "prompt": json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"],
        }
        if "reward_model" in row:
            rm = json.loads(row["reward_model"]) if isinstance(row["reward_model"], str) else row["reward_model"]
            record["ground_truth"] = rm.get("ground_truth", "")
        if "extra_info" in row:
            record["extra_info"] = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
        records.append(record)

    if max_samples > 0:
        records = records[:max_samples]
    return records


def load_dataset_from_json(json_path: str, max_samples: int = -1):
    """Load evaluation data from a JSON or JSONL file.

    Expected format per record:
      {"prompt": [{"role": "user", "content": "..."}], "ground_truth": "...", "data_source": "..."}
    or simply:
      {"prompt": "plain text question", ...}
    """
    records = []
    if json_path.endswith(".jsonl"):
        with open(json_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(json_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                records = [data]

    # Normalize: ensure prompt is chat format
    for r in records:
        if isinstance(r["prompt"], str):
            r["prompt"] = [{"role": "user", "content": r["prompt"]}]

    if max_samples > 0:
        records = records[:max_samples]
    return records


def generate_with_logprobs(
    model_path: str,
    prompts: list,
    n_samples: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    top_logprobs: int = 50,
):
    """Generate responses with per-token log probabilities using vLLM.

    Returns list of dicts with token-level info for each (prompt, sample).
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    formatted_prompts = []
    for p in prompts:
        if isinstance(p, list):
            text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        else:
            text = p
        formatted_prompts.append(text)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p if temperature > 0 else 1.0,
        max_tokens=max_tokens,
        logprobs=top_logprobs,
    )

    outputs = llm.generate(formatted_prompts, sampling_params)

    all_results = []
    for prompt_idx, output in enumerate(outputs):
        for sample_idx, completion in enumerate(output.outputs):
            tokens_data = []
            if completion.logprobs:
                for pos, logprob_dict in enumerate(completion.logprobs):
                    # logprob_dict: dict mapping token_id -> LogprobsOutput
                    # The generated token's info
                    token_id = completion.token_ids[pos] if pos < len(completion.token_ids) else None
                    token_str = None
                    token_logprob = None

                    # Collect top logprobs for entropy computation
                    top_logprobs_list = []
                    for tid, lp_info in logprob_dict.items():
                        top_logprobs_list.append({
                            "token_id": tid,
                            "token": lp_info.decoded_token if hasattr(lp_info, 'decoded_token') else str(tid),
                            "logprob": lp_info.logprob,
                        })
                        if tid == token_id:
                            token_str = lp_info.decoded_token if hasattr(lp_info, 'decoded_token') else str(tid)
                            token_logprob = lp_info.logprob

                    # If we didn't find the generated token in the dict, use the first entry
                    if token_str is None and len(top_logprobs_list) > 0:
                        token_str = top_logprobs_list[0]["token"]
                        token_logprob = top_logprobs_list[0]["logprob"]

                    # Compute neg_log_prob
                    neg_log_prob = -token_logprob if token_logprob is not None else 0.0

                    # Compute entropy from top logprobs (approximate)
                    # H = -sum P(v) * log P(v)
                    entropy = compute_entropy_from_logprobs(top_logprobs_list)

                    tokens_data.append({
                        "position": pos,
                        "token_id": token_id,
                        "token": token_str,
                        "logprob": token_logprob,
                        "neg_log_prob": neg_log_prob,
                        "entropy": entropy,
                    })

            result = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "text": completion.text,
                "num_tokens": len(tokens_data),
                "tokens": tokens_data,
            }
            all_results.append(result)

    return all_results


def compute_entropy_from_logprobs(top_logprobs_list: list) -> float:
    """Compute entropy from a list of top logprobs.

    H = -sum_i P(v_i) * log P(v_i)

    Since we only have top-k logprobs, we compute a lower-bound entropy
    over the observed tokens. For a better approximation, we distribute
    remaining probability mass uniformly (if sum < 1).
    """
    if not top_logprobs_list:
        return 0.0

    logprobs = np.array([lp["logprob"] for lp in top_logprobs_list])
    probs = np.exp(logprobs)

    # Entropy from observed tokens
    # H = -sum p * log(p) = -sum p * logprob
    entropy = -np.sum(probs * logprobs)

    # Account for remaining probability mass
    remaining_prob = max(0.0, 1.0 - np.sum(probs))
    if remaining_prob > 1e-10:
        # Assume remaining mass has at least as high entropy as observed
        # Use uniform assumption: remaining mass contributes remaining_prob * (-log(remaining_prob))
        entropy += -remaining_prob * np.log(remaining_prob + 1e-30)

    return float(entropy)


def generate_with_logprobs_hf(
    model_path: str,
    prompts: list,
    n_samples: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    device: str = "auto",
):
    """Fallback: generate with HuggingFace transformers (slower, but no vLLM needed).

    Computes full vocabulary entropy (exact).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    all_results = []

    for prompt_idx, prompt in enumerate(prompts):
        if isinstance(prompt, list):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

        for sample_idx in range(n_samples):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = output.sequences[0][input_ids.shape[1]:]
            scores = output.scores  # tuple of (vocab_size,) tensors

            tokens_data = []
            for pos in range(len(generated_ids)):
                token_id = generated_ids[pos].item()
                token_str = tokenizer.decode([token_id])

                # scores[pos] shape: (1, vocab_size) or (vocab_size,)
                logits = scores[pos]
                if logits.dim() == 2:
                    logits = logits[0]

                # Compute log probs from logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probs = torch.exp(log_probs)

                token_logprob = log_probs[token_id].item()
                neg_log_prob = -token_logprob

                # Exact entropy over full vocabulary
                entropy = -(probs * log_probs).sum().item()
                # Handle NaN from 0 * log(0)
                if np.isnan(entropy):
                    entropy = 0.0

                tokens_data.append({
                    "position": pos,
                    "token_id": token_id,
                    "token": token_str,
                    "logprob": token_logprob,
                    "neg_log_prob": neg_log_prob,
                    "entropy": entropy,
                })

            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_results.append({
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "text": response_text,
                "num_tokens": len(tokens_data),
                "tokens": tokens_data,
            })

        print(f"  [{prompt_idx+1}/{len(prompts)}] Generated {n_samples} sample(s)")

    return all_results


def run_analysis(args):
    """Main analysis pipeline."""
    # Load data
    if args.dataset_path.endswith(".parquet"):
        records = load_dataset_from_parquet(args.dataset_path, args.max_samples)
    else:
        records = load_dataset_from_json(args.dataset_path, args.max_samples)

    print(f"Loaded {len(records)} samples from {args.dataset_path}")

    prompts = [r["prompt"] for r in records]

    # Generate with logprobs
    print(f"Generating responses with {args.model_path} ...")
    start_time = time.time()

    if args.backend == "vllm":
        results = generate_with_logprobs(
            model_path=args.model_path,
            prompts=prompts,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            top_logprobs=args.top_logprobs,
        )
    else:
        results = generate_with_logprobs_hf(
            model_path=args.model_path,
            prompts=prompts,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )

    elapsed = time.time() - start_time
    print(f"Generation complete in {elapsed:.1f}s")

    # Enrich results with prompt info
    for result in results:
        pidx = result["prompt_idx"]
        record = records[pidx]
        result["prompt"] = record["prompt"]
        result["data_source"] = record.get("data_source", "unknown")
        result["ground_truth"] = record.get("ground_truth", "")

        # Compute summary statistics for this response
        if result["tokens"]:
            neg_lps = [t["neg_log_prob"] for t in result["tokens"]]
            entropies = [t["entropy"] for t in result["tokens"]]
            result["summary"] = {
                "neg_log_prob_mean": float(np.mean(neg_lps)),
                "neg_log_prob_std": float(np.std(neg_lps)),
                "neg_log_prob_max": float(np.max(neg_lps)),
                "neg_log_prob_min": float(np.min(neg_lps)),
                "neg_log_prob_median": float(np.median(neg_lps)),
                "entropy_mean": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies)),
                "entropy_max": float(np.max(entropies)),
                "entropy_min": float(np.min(entropies)),
                "entropy_median": float(np.median(entropies)),
            }

    # Save results
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if args.output_path.endswith(".json"):
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        # JSONL format
        with open(args.output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output_path}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    all_neg_lps = []
    all_entropies = []
    for r in results:
        for t in r["tokens"]:
            all_neg_lps.append(t["neg_log_prob"])
            all_entropies.append(t["entropy"])

    if all_neg_lps:
        all_neg_lps = np.array(all_neg_lps)
        all_entropies = np.array(all_entropies)
        print(f"Total tokens analyzed: {len(all_neg_lps)}")
        print(f"-log_prob:  mean={all_neg_lps.mean():.4f}, std={all_neg_lps.std():.4f}, "
              f"median={np.median(all_neg_lps):.4f}, max={all_neg_lps.max():.4f}")
        print(f"Entropy:    mean={all_entropies.mean():.4f}, std={all_entropies.std():.4f}, "
              f"median={np.median(all_entropies):.4f}, max={all_entropies.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Reasoning Analysis: per-token log_prob and entropy")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HuggingFace or local)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to evaluation data (.parquet, .json, or .jsonl)")
    parser.add_argument("--output_path", type=str, default="reasoning_analysis/outputs/analysis.jsonl",
                        help="Output file path (.json or .jsonl)")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max number of prompts to evaluate (-1 = all)")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of response samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--top_logprobs", type=int, default=50,
                        help="Number of top logprobs to request from vLLM (for entropy approximation)")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"],
                        help="Inference backend: vllm (fast, approximate entropy) or hf (slow, exact entropy)")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
