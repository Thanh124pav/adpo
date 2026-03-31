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

# Add project root to path so we can import reward functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adpo.reward_functions import compute_score


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (ndarray, int64, float64, etc.)."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def load_dataset_from_parquet(parquet_path: str, max_samples: int = -1):
    """Load evaluation data from parquet file.

    Handles both JSON-string columns (from prepare_datasets.py) and native
    numpy array columns (verl's parquet format).
    """
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        # Parse prompt — JSON string or numpy array (verl format)
        raw_prompt = row["prompt"]
        if isinstance(raw_prompt, str):
            prompt = json.loads(raw_prompt)
        else:
            prompt = raw_prompt  # keep as-is (numpy array is list-like)

        record = {
            "data_source": row.get("data_source", "unknown"),
            "prompt": prompt,
        }
        if "reward_model" in row:
            raw_rm = row["reward_model"]
            rm = json.loads(raw_rm) if isinstance(raw_rm, str) else raw_rm
            record["ground_truth"] = rm.get("ground_truth", "") if hasattr(rm, "get") else ""
        if "extra_info" in row:
            raw_ei = row["extra_info"]
            record["extra_info"] = json.loads(raw_ei) if isinstance(raw_ei, str) else raw_ei
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
    top_logprobs: int = 20,
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
        if isinstance(p, (list, np.ndarray)):
            # numpy array (verl parquet) or list — both are valid chat format
            text = tokenizer.apply_chat_template(
                list(p) if isinstance(p, np.ndarray) else p,
                tokenize=False, add_generation_prompt=True,
            )
        else:
            text = str(p)
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

    # Free vLLM GPU memory so subsequent HF model loads don't OOM
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return all_results


def compute_entropy_from_logprobs(top_logprobs_list: list) -> float:
    """Compute approximate entropy from a list of top logprobs.

    H = -sum_i P(v_i) * log P(v_i)

    Since we only have top-k logprobs, this is an approximation.
    Remaining probability mass is accounted for with a uniform assumption.
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
        entropy += -remaining_prob * np.log(remaining_prob + 1e-30)

    return float(entropy)


# ---------------------------------------------------------------------------
# Exact Entropy via Forward Pass
# ---------------------------------------------------------------------------


def compute_exact_entropy_forward_pass(
    model_path: str,
    results: list,
    prompts: list,
    batch_size: int = 4,
    device: str = "auto",
):
    """Compute exact entropy by running a forward pass over generated sequences.

    After vLLM generation, this function loads the model in HuggingFace,
    concatenates [prompt + generated_tokens], runs a forward pass to get
    full logits at each generated token position, and computes:
      - exact_entropy: H(t) = -sum_v P(v|context_t) * log P(v|context_t)
      - exact_neg_log_prob: -log P(token_t | context_t) from full distribution

    This replaces the approximate values from top-k logprobs.

    Args:
        model_path: HuggingFace model path.
        results: List of result dicts from generate_with_logprobs().
        prompts: Original prompts (chat format or string).
        batch_size: Batch size for forward pass.
        device: Device for HuggingFace model.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading model for exact entropy forward pass ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i, result in enumerate(results):
        prompt_idx = result["prompt_idx"]
        prompt = prompts[prompt_idx]

        # Build the full sequence: prompt + response
        if isinstance(prompt, (list, np.ndarray)):
            prompt_text = tokenizer.apply_chat_template(
                list(prompt) if isinstance(prompt, np.ndarray) else prompt,
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = str(prompt)

        # Tokenize prompt to know where response starts
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Build full sequence: prompt + generated token ids
        gen_token_ids = [t["token_id"] for t in result["tokens"]]
        if not gen_token_ids:
            continue

        full_ids = prompt_ids + gen_token_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long).to(model.device)

        # Forward pass to get logits at every position
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # For each generated token at position t in the response,
        # the logits that predict it are at position (prompt_len + t - 1)
        # because logits[i] predicts token[i+1]
        for t_idx, token_data in enumerate(result["tokens"]):
            logit_pos = prompt_len + t_idx - 1
            if logit_pos < 0 or logit_pos >= logits.shape[1]:
                token_data["exact_entropy"] = 0.0
                token_data["exact_neg_log_prob"] = token_data["neg_log_prob"]
                token_data["entropy_method"] = "fallback"
                continue

            token_logits = logits[0, logit_pos, :]  # (vocab_size,)

            # Compute log_softmax for numerical stability
            log_probs = torch.nn.functional.log_softmax(token_logits.float(), dim=-1)
            probs = torch.exp(log_probs)

            # Exact entropy: H = -sum_v P(v) * log P(v)
            # Use nansum to handle 0 * log(0) = 0
            entropy_terms = probs * log_probs
            entropy_terms = torch.nan_to_num(entropy_terms, nan=0.0)
            exact_entropy = -entropy_terms.sum().item()

            # Exact neg_log_prob for the generated token
            token_id = token_data["token_id"]
            exact_logprob = log_probs[token_id].item()
            exact_neg_log_prob = -exact_logprob

            token_data["exact_entropy"] = float(exact_entropy)
            token_data["exact_neg_log_prob"] = float(exact_neg_log_prob)
            token_data["approx_entropy"] = token_data["entropy"]
            token_data["approx_neg_log_prob"] = token_data["neg_log_prob"]
            # Replace main fields with exact values
            token_data["entropy"] = float(exact_entropy)
            token_data["neg_log_prob"] = float(exact_neg_log_prob)
            token_data["logprob"] = float(exact_logprob)
            token_data["entropy_method"] = "exact"

        if (i + 1) % 10 == 0 or i == len(results) - 1:
            print(f"  Forward pass: [{i+1}/{len(results)}] responses processed")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Exact entropy computation complete.")


# ---------------------------------------------------------------------------
# Attention Matrix & Hidden States Extraction
# ---------------------------------------------------------------------------


def _find_think_boundary(token_ids: list, tokenizer) -> int | None:
    """Find position after </think> in generated token ids.

    Returns index (relative to token_ids) right after </think>,
    or None if not found.
    """
    # Try single-token match first (Qwen3 treats </think> as special token)
    think_end_candidates = []
    for text in ("</think>", "<|/think|>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            think_end_candidates.append(ids)

    # Also check tokenizer's special tokens
    if hasattr(tokenizer, "added_tokens_encoder"):
        for tok_str, tok_id in tokenizer.added_tokens_encoder.items():
            if "think" in tok_str.lower() and ("/" in tok_str or "end" in tok_str.lower()):
                think_end_candidates.append([tok_id])

    for end_ids in think_end_candidates:
        end_len = len(end_ids)
        for j in range(len(token_ids) - end_len + 1):
            if token_ids[j:j + end_len] == end_ids:
                return j + end_len

    return None


def extract_attention_hidden_states(
    model_path: str,
    results: list,
    prompts: list,
    output_dir: str,
    device: str = "auto",
    attn_impl: str = "flash_attention_2",
):
    """Extract hidden states via forward pass.

    Attention matrices are NOT saved — they will be reconstructed
    from hidden states + model weights at visualization time.

    Saves one .npz file per response in output_dir.

    Args:
        model_path: HuggingFace model path.
        results: List of result dicts from generate_with_logprobs().
        prompts: Original prompts (chat format or string).
        output_dir: Directory to save .npz files.
        device: Device for model.
        attn_impl: Attention implementation ('flash_attention_2' or 'eager').
                   flash_attention_2 is default and faster since we only need
                   hidden states, not attention weights.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model for hidden states extraction (attn={attn_impl}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        prompt_idx = result["prompt_idx"]
        prompt = prompts[prompt_idx]

        # Build full sequence: prompt + generated tokens
        if isinstance(prompt, (list, np.ndarray)):
            prompt_text = tokenizer.apply_chat_template(
                list(prompt) if isinstance(prompt, np.ndarray) else prompt,
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text = str(prompt)

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        gen_token_ids = [t["token_id"] for t in result["tokens"]]
        if not gen_token_ids:
            continue

        full_ids = prompt_ids + gen_token_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long).to(model.device)

        # Forward pass — only need hidden states (no attention needed)
        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=False,
                output_hidden_states=True,
            )

        # --- Determine thinking / output boundary ---
        think_boundary = _find_think_boundary(gen_token_ids, tokenizer)

        gen_start = prompt_len  # absolute position in full sequence
        gen_end = len(full_ids)

        if think_boundary is not None:
            think_len = think_boundary
            out_len = (gen_end - gen_start) - think_boundary
        else:
            think_len = 0
            out_len = gen_end - gen_start

        # --- Extract hidden states ---
        # outputs.hidden_states: tuple of (1, seq_len, hidden_dim) per layer + embedding
        # hidden_states[i] = input to layer i (i=0 is embedding output)
        hidden_list = []
        for layer_idx in range(len(outputs.hidden_states)):
            hs = outputs.hidden_states[layer_idx][0]  # (seq_len, hidden_dim)
            # Only keep response (thinking + output) token positions
            hs_response = hs[gen_start:gen_end].cpu().to(torch.float16).numpy()
            hidden_list.append(hs_response)

        # --- Save to .npz ---
        npz_data = {
            # (all_layers+1, response_len, hidden_dim) in float16
            "hidden_states": np.stack(hidden_list),
        }

        # Metadata
        metadata = {
            "prompt_idx": int(prompt_idx),
            "sample_idx": int(result["sample_idx"]),
            "prompt_len": prompt_len,
            "gen_len": len(gen_token_ids),
            "think_boundary": think_boundary,
            "think_len": think_len,
            "out_len": out_len,
            "n_model_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "num_kv_heads": getattr(model.config, "num_key_value_heads",
                                     model.config.num_attention_heads),
            "correct": result.get("correct"),
            "score": result.get("score"),
        }
        npz_data["metadata"] = np.array(json.dumps(metadata))

        # Tokens text for reference
        gen_tokens_str = [t["token"] for t in result["tokens"]]
        npz_data["tokens"] = np.array(gen_tokens_str)

        npz_path = os.path.join(output_dir, f"response_{i:04d}.npz")
        np.savez_compressed(npz_path, **npz_data)

        # Free memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or i == len(results) - 1:
            print(f"  [{i+1}/{len(results)}] Saved {npz_path}")

        size_kb = os.path.getsize(npz_path) / 1024
        print(f"  [{i+1}/{len(results)}] {npz_path} ({size_kb:.0f} KB) "
              f"think={think_len} tokens, out={out_len} tokens"
              f"{', no </think> found' if think_boundary is None else ''}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Extraction complete. Files saved to {output_dir}/")


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

    # Ensure vLLM GPU memory is fully released before loading HF models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Exact entropy via forward pass (overrides approximate values from vLLM top-k)
    if args.exact_entropy:
        print("\nComputing exact entropy via forward pass ...")
        start_time2 = time.time()
        compute_exact_entropy_forward_pass(
            model_path=args.model_path,
            results=results,
            prompts=prompts,
            device=args.device,
        )
        elapsed2 = time.time() - start_time2
        print(f"Exact entropy computation done in {elapsed2:.1f}s")

    # Enrich results with prompt info and accuracy
    for result in results:
        pidx = result["prompt_idx"]
        record = records[pidx]
        result["prompt"] = record["prompt"]
        result["data_source"] = record.get("data_source", "unknown")
        result["ground_truth"] = record.get("ground_truth", "")

        # Compute accuracy using the reward function
        score = compute_score(
            data_source=result["data_source"],
            solution_str=result["text"],
            ground_truth=result["ground_truth"],
            extra_info=record.get("extra_info"),
        )
        result["score"] = score
        result["correct"] = score >= 1.0

        # Compute summary statistics for this response
        if result["tokens"]:
            neg_lps = [t["neg_log_prob"] for t in result["tokens"]]
            entropies = [t["entropy"] for t in result["tokens"]]
            summary = {
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
            # If exact entropy was computed, also store approximate stats
            if "approx_entropy" in result["tokens"][0]:
                approx_ents = [t["approx_entropy"] for t in result["tokens"]]
                approx_nlps = [t["approx_neg_log_prob"] for t in result["tokens"]]
                summary["approx_entropy_mean"] = float(np.mean(approx_ents))
                summary["approx_neg_log_prob_mean"] = float(np.mean(approx_nlps))
                summary["entropy_method"] = "exact"
            else:
                summary["entropy_method"] = "approximate" if args.backend == "vllm" else "exact"
            result["summary"] = summary

    # Save results
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if args.output_path.endswith(".json"):
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    else:
        # JSONL format
        with open(args.output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False, cls=_NumpyEncoder) + "\n")

    print(f"Saved {len(results)} results to {args.output_path}")

    # Extract hidden states (attention reconstructed later from these)
    if args.extract_internals:
        print("\nExtracting hidden states ...")
        internals_dir = os.path.join(
            os.path.dirname(args.output_path) or ".", "internals"
        )
        start_time3 = time.time()
        extract_attention_hidden_states(
            model_path=args.model_path,
            results=results,
            prompts=prompts,
            output_dir=internals_dir,
            device=args.device,
            attn_impl=args.attn_impl,
        )
        elapsed3 = time.time() - start_time3
        print(f"Extraction done in {elapsed3:.1f}s")

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

    # Accuracy
    n_correct = sum(1 for r in results if r.get("correct", False))
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print(f"Accuracy: {n_correct}/{n_total} = {accuracy:.4f} ({accuracy*100:.1f}%)")

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
    parser.add_argument("--top_logprobs", type=int, default=20,
                        help="Number of top logprobs to request from vLLM (max 20, for entropy approximation)")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"],
                        help="Inference backend: vllm (fast, approximate entropy) or hf (slow, exact entropy)")
    parser.add_argument("--exact_entropy", action="store_true",
                        help="Compute exact entropy via forward pass after vLLM generation. "
                             "Loads the model in HF transformers and runs a forward pass over "
                             "each [prompt + response] to get full logit distribution at every "
                             "token position. Slower but gives exact entropy values.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for HF model in exact_entropy mode (default: auto)")
    parser.add_argument("--extract_internals", action="store_true",
                        help="Extract hidden states via forward pass. Saves .npz files per response. "
                             "Attention matrices are reconstructed at visualization time from "
                             "hidden states + model weights (no need to store them).")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "eager"],
                        help="Attention implementation for hidden states extraction. "
                             "flash_attention_2 (default) is faster; eager is slower but "
                             "compatible with all GPUs.")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
