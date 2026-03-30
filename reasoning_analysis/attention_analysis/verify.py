"""
Verify attention reconstruction from hidden states.

Loads a model, runs forward pass with output_attentions=True and
output_hidden_states=True, then reconstructs attention from hidden
states and compares with ground truth.

Usage:
    python reasoning_analysis/attention_analysis/verify.py \
        --model_path Qwen/Qwen3-0.6B --num_samples 10

    python reasoning_analysis/attention_analysis/verify.py \
        --model_path Qwen/Qwen2.5-0.5B --num_samples 10
"""

import argparse
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_analysis.reconstruct import reconstruct_attention


# ---------------------------------------------------------------------------
# Test prompts (simple, diverse lengths)
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "What is 2+2?",
    "Explain gravity in one sentence.",
    "Hello, how are you today?",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
    "List three prime numbers.",
    "Why is the sky blue?",
    "Translate 'hello' to Spanish.",
    "What is machine learning?",
    "Count from 1 to 5.",
    "Who invented the telephone?",
    "What color is the sun?",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt: torch.Tensor, recon: torch.Tensor) -> dict:
    """Compare ground-truth and reconstructed attention matrices.

    Args:
        gt: (num_heads, seq, seq) ground-truth attention weights (float32).
        recon: (num_heads, seq, seq) reconstructed attention weights (float32).

    Returns:
        dict with MSE, max_abs_error, cosine_sim, kl_divergence.
    """
    gt = gt.float()
    recon = recon.float()

    # MSE
    mse = ((gt - recon) ** 2).mean().item()

    # Max absolute error
    max_abs = (gt - recon).abs().max().item()

    # Cosine similarity (per attention row, then averaged)
    # Flatten each head's rows
    gt_flat = gt.reshape(-1, gt.shape[-1])      # (num_heads * seq, seq)
    recon_flat = recon.reshape(-1, recon.shape[-1])
    cos_sim = torch.nn.functional.cosine_similarity(gt_flat, recon_flat, dim=-1)
    cos_sim_mean = cos_sim.mean().item()

    # KL divergence: KL(gt || recon), averaged over all rows
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    gt_safe = gt_flat.clamp(min=eps)
    recon_safe = recon_flat.clamp(min=eps)
    kl = (gt_safe * (gt_safe.log() - recon_safe.log())).sum(dim=-1).mean().item()

    return {
        "mse": mse,
        "max_abs_error": max_abs,
        "cosine_sim": cos_sim_mean,
        "kl_divergence": kl,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_model(model_path: str, num_samples: int, device: str = "auto"):
    """Run full verification for one model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"  Verifying: {model_path}")
    print(f"  Samples: {num_samples}")
    print(f"{'='*70}\n")

    # Load model with eager attention (required for output_attentions)
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    n_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    print(f"  Architecture: {config.model_type}")
    print(f"  Layers: {n_layers}, Heads: {num_heads}, KV Heads: {num_kv_heads}")
    print(f"  Hidden dim: {config.hidden_size}, Head dim: {config.hidden_size // num_heads}")
    has_qk_norm = hasattr(model.model.layers[0].self_attn, "q_norm") and \
                  model.model.layers[0].self_attn.q_norm is not None
    print(f"  QK-Norm: {has_qk_norm}")
    print()

    # Prepare test inputs
    prompts = TEST_PROMPTS[:num_samples]

    # Select layers to verify: first, middle, last
    verify_layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                3 * n_layers // 4, n_layers - 1]))
    print(f"Verifying layers: {verify_layers}\n")

    # Collect per-layer metrics across all samples
    all_metrics = {layer: [] for layer in verify_layers}

    for sample_idx, prompt_text in enumerate(prompts):
        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt_text

        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)

        # Forward pass: get both ground-truth attention and hidden states
        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
            )

        print(f"Sample {sample_idx}: \"{prompt_text[:50]}\" (seq_len={seq_len})")

        for layer_idx in verify_layers:
            # Ground truth: (1, num_heads, seq, seq) -> (num_heads, seq, seq)
            gt_attn = outputs.attentions[layer_idx][0].float()

            # Reconstruct from hidden states
            # hidden_states[layer_idx] = input to layer layer_idx
            hidden_state = outputs.hidden_states[layer_idx].unsqueeze(0) \
                if outputs.hidden_states[layer_idx].dim() == 2 \
                else outputs.hidden_states[layer_idx]

            recon_attn = reconstruct_attention(
                model, layer_idx, hidden_state, position_ids
            )

            # Compare
            metrics = compute_metrics(gt_attn, recon_attn)
            all_metrics[layer_idx].append(metrics)

            status = "OK" if metrics["mse"] < 1e-5 else "WARN" if metrics["mse"] < 1e-3 else "FAIL"
            print(f"  Layer {layer_idx:3d}: MSE={metrics['mse']:.2e}  "
                  f"MaxErr={metrics['max_abs_error']:.2e}  "
                  f"CosSim={metrics['cosine_sim']:.6f}  "
                  f"KL={metrics['kl_divergence']:.2e}  [{status}]")

        # Free memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Layer':>8} {'MSE (mean)':>12} {'MSE (max)':>12} "
          f"{'CosSim':>10} {'KL':>12} {'Status':>8}")
    print("-" * 70)

    all_pass = True
    for layer_idx in verify_layers:
        metrics_list = all_metrics[layer_idx]
        mse_mean = np.mean([m["mse"] for m in metrics_list])
        mse_max = np.max([m["mse"] for m in metrics_list])
        cos_mean = np.mean([m["cosine_sim"] for m in metrics_list])
        kl_mean = np.mean([m["kl_divergence"] for m in metrics_list])

        if mse_max < 1e-5:
            status = "PASS"
        elif mse_max < 1e-3:
            status = "WARN"
        else:
            status = "FAIL"
            all_pass = False

        print(f"{layer_idx:>8} {mse_mean:>12.2e} {mse_max:>12.2e} "
              f"{cos_mean:>10.6f} {kl_mean:>12.2e} {status:>8}")

    print("-" * 70)
    if all_pass:
        print(f"  RESULT: ALL LAYERS PASSED for {model_path}")
    else:
        print(f"  RESULT: SOME LAYERS FAILED for {model_path}")
    print(f"  (threshold: MSE < 1e-5 = PASS, < 1e-3 = WARN, else FAIL)")
    print(f"{'='*70}\n")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify attention reconstruction from hidden states"
    )
    parser.add_argument(
        "--model_path", type=str, nargs="+",
        default=["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
        help="Model(s) to verify (default: Qwen3-0.6B and Qwen2.5-0.5B)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of test samples per model"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (default: auto)"
    )
    args = parser.parse_args()

    results = {}
    for model_path in args.model_path:
        try:
            passed = verify_model(model_path, args.num_samples, args.device)
            results[model_path] = passed
        except Exception as e:
            print(f"\nERROR verifying {model_path}: {e}")
            import traceback
            traceback.print_exc()
            results[model_path] = False

    # Final report
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    for model_path, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {model_path}")
    print("=" * 70)

    all_ok = all(results.values())
    if all_ok:
        print("  All models passed reconstruction verification!")
    else:
        print("  Some models failed. Check output above for details.")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
