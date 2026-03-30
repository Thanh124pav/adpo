#!/usr/bin/env python3
"""Merge verl FSDP checkpoint → HuggingFace model (full or LoRA-merged).

Steps:
1. Use verl's model_merger to convert FSDP shards → HF format (or LoRA adapter)
2. If LoRA adapter is produced, merge it into the base model
3. Save final merged model ready for vLLM serving

Usage:
    # Auto-detect base model from config.json in checkpoint
    python scripts/merge.py \
        --checkpoint checkpoints/experiment/global_step_250/actor \
        --output checkpoints/hf_merged/step_250

    # Specify base model explicitly
    python scripts/merge.py \
        --checkpoint checkpoints/experiment/global_step_250/actor \
        --output checkpoints/hf_merged/step_250 \
        --base_model Qwen/Qwen3-4B

    # Skip verl merge (already have adapter), just merge LoRA
    python scripts/merge.py \
        --adapter checkpoints/hf_merged/step_250/lora_adapter \
        --output checkpoints/hf_merged/step_250_full \
        --base_model Qwen/Qwen3-4B
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def find_base_model(checkpoint_dir):
    """Try to find base model path from checkpoint's config.json."""
    hf_dir = os.path.join(checkpoint_dir, "huggingface")
    config_path = os.path.join(hf_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        # Some configs store _name_or_path
        model_path = config.get("_name_or_path", "")
        if model_path and model_path != "":
            return model_path
    return None


def step1_verl_merge(checkpoint_dir, intermediate_dir):
    """Convert FSDP shards to HF format using verl's model_merger."""
    print(f"\n[Step 1] Converting FSDP shards → HF format")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Output: {intermediate_dir}")

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", checkpoint_dir,
        "--target_dir", intermediate_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] verl model_merger failed with code {result.returncode}")
        sys.exit(1)

    # Check what was produced
    adapter_dir = os.path.join(intermediate_dir, "lora_adapter")
    if os.path.exists(adapter_dir):
        print(f"  → LoRA adapter saved to {adapter_dir}")
        return adapter_dir

    # Check if full model was saved (has .safetensors or .bin)
    for f in os.listdir(intermediate_dir):
        if f.endswith((".safetensors", ".bin")):
            print(f"  → Full model saved to {intermediate_dir}")
            return None  # No adapter, full model already merged

    print(f"  [WARN] No model weights found in {intermediate_dir}")
    return None


def step2_merge_lora(adapter_dir, base_model, output_dir):
    """Merge LoRA adapter into base model."""
    print(f"\n[Step 2] Merging LoRA adapter into base model")
    print(f"  Base model: {base_model}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Output: {output_dir}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", trust_remote_code=True,
    )

    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    print("  Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Verify
    n_files = len([f for f in os.listdir(output_dir) if f.endswith((".safetensors", ".bin"))])
    print(f"  → Done! {n_files} weight file(s) saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge verl FSDP checkpoint to HuggingFace model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to verl checkpoint actor/ dir (contains model_world_size_*.pt)")
    parser.add_argument("--adapter", default=None,
                        help="Path to existing LoRA adapter dir (skip verl merge step)")
    parser.add_argument("--base_model", default=None,
                        help="Base model path/name (auto-detected from checkpoint if not specified)")
    parser.add_argument("--output", required=True,
                        help="Output directory for merged HF model")
    args = parser.parse_args()

    if args.checkpoint is None and args.adapter is None:
        parser.error("Either --checkpoint or --adapter must be specified")

    adapter_dir = args.adapter

    # Step 1: verl FSDP → HF (if checkpoint provided)
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

        # Use output dir as intermediate if no adapter specified
        intermediate_dir = args.output if adapter_dir else args.output
        adapter_dir = step1_verl_merge(args.checkpoint, intermediate_dir)

    # If no LoRA adapter was produced, we're done (full model already saved)
    if adapter_dir is None:
        print(f"\n[Done] Full model saved to {args.output}")
        return

    # Step 2: Merge LoRA into base model
    base_model = args.base_model
    if base_model is None and args.checkpoint is not None:
        base_model = find_base_model(args.checkpoint)
    if base_model is None:
        print("[ERROR] Cannot determine base model. Specify --base_model explicitly.")
        print("  Example: --base_model Qwen/Qwen3-4B")
        sys.exit(1)

    # If output == intermediate (verl wrote adapter there), use a temp dir then replace
    if os.path.exists(os.path.join(args.output, "lora_adapter")):
        merged_dir = args.output + "_merged"
        step2_merge_lora(adapter_dir, base_model, merged_dir)
        # Move merged model back to output dir
        # Keep lora_adapter as backup
        for f in os.listdir(merged_dir):
            src = os.path.join(merged_dir, f)
            dst = os.path.join(args.output, f)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
            else:
                shutil.move(src, dst)
        shutil.rmtree(merged_dir)
    else:
        step2_merge_lora(adapter_dir, base_model, args.output)

    print(f"\n[Done] Merged model ready at {args.output}")
    print(f"  Serve with: vllm serve {args.output}")


if __name__ == "__main__":
    main()
