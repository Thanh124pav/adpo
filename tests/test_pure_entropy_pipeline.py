"""
End-to-end pipeline test for pure-entropy algorithm.

Simulates the full training flow with a real model:
1. Tokenize sample prompts + responses
2. Compute entropy → detect phase boundaries
3. Partial forward (hook) → get hidden states at layer L
4. Reconstruct attention → build phase attention matrix
5. Build influence matrix A → solve rewards
6. Compute advantages (local + global)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_pure_entropy_pipeline.py \
        --model /path/to/model

    # Shorter test (2 samples instead of 8):
    CUDA_VISIBLE_DEVICES=0 python tests/test_pure_entropy_pipeline.py \
        --model /path/to/model --batch-size 2
"""

import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, ".")


SAMPLE_RESPONSES = [
    # Correct answer with thinking
    "<think>\nLet me solve 2+3 step by step.\nFirst, I start with 2.\n"
    "Then I add 3 to get 5.\nLet me verify: 2+3 = 5. Yes.\n</think>\n"
    "The answer is \\boxed{5}.",

    # Incorrect answer
    "<think>\nI need to find 2+3.\nHmm, 2+3 = 6? No wait.\n"
    "Actually 2+3 = 4. Let me check again.\n</think>\n"
    "The answer is \\boxed{4}.",

    # Long reasoning, correct
    "<think>\nTo compute 10*5, I can think of it as 10 groups of 5.\n"
    "That gives 50. Or equivalently, 5*10 = 50.\n"
    "Let me double check: 10+10+10+10+10 = 50. Confirmed.\n</think>\n"
    "The answer is \\boxed{50}.",

    # Short response, correct
    "<think>\n1+1=2.\n</think>\nThe answer is \\boxed{2}.",

    # Medium reasoning, incorrect
    "<think>\nWhat is 7*8?\n7*8 = 7*7 + 7 = 49 + 7 = 56.\n"
    "Wait, let me recalculate. 7*8 = 54? No.\n"
    "7*8 = 56. Yes that's right.\n</think>\n"
    "The answer is \\boxed{54}.",

    # Very long reasoning
    "<think>\nI need to find the sum 1+2+3+...+10.\n"
    "Using the formula n*(n+1)/2 with n=10:\n"
    "10*11/2 = 110/2 = 55.\n"
    "Alternatively, I can pair: (1+10)+(2+9)+(3+8)+(4+7)+(5+6) = 5*11 = 55.\n"
    "Both methods give 55.\n</think>\n"
    "The answer is \\boxed{55}.",

    # Minimal thinking
    "<think>\n3*3=9.\n</think>\n\\boxed{9}.",

    # Multi-step
    "<think>\nStep 1: 2^3 = 8.\nStep 2: 8 + 2 = 10.\n"
    "Step 3: 10 / 2 = 5.\nSo 2^3 + 2 divided by 2 is 5.\n</think>\n"
    "The answer is \\boxed{5}.",
]

GROUND_TRUTHS = ["5", "5", "50", "2", "56", "55", "9", "5"]


def run_pipeline_test(model_path: str, batch_size: int = 8):
    """Run full end-to-end pipeline test."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from adpo.pure_entropy_algorithm import (
        compute_token_entropy,
        detect_phase_boundaries_pure_entropy,
        segment_response_into_phases,
        _partial_forward,
        reconstruct_attention_at_layer,
        build_phase_attention_matrix,
        build_influence_matrix_A,
        solve_phase_rewards,
        compute_pure_entropy_advantages,
    )
    from adpo.reward_functions import compute_score

    batch_size = min(batch_size, len(SAMPLE_RESPONSES))
    responses = SAMPLE_RESPONSES[:batch_size]
    ground_truths = GROUND_TRUTHS[:batch_size]

    # =========================================================================
    # Step 0: Load model + tokenizer
    # =========================================================================
    print(f"{'='*70}")
    print(f"PURE-ENTROPY PIPELINE TEST")
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    print(f"\n[Step 0] Model loaded: {n_layers} layers, device={device} ({time.time()-t0:.1f}s)")

    # =========================================================================
    # Step 1: Tokenize → build input_ids + response_mask
    # =========================================================================
    print(f"\n[Step 1] Tokenizing {batch_size} responses...")

    # Simple prompt prefix
    prompt = "Solve the math problem.\n\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    all_ids = []
    all_resp_masks = []
    max_len = 0

    for resp_text in responses:
        resp_ids = tokenizer.encode(resp_text, add_special_tokens=False)
        full_ids = prompt_ids + resp_ids
        mask = [0] * prompt_len + [1] * len(resp_ids)
        all_ids.append(full_ids)
        all_resp_masks.append(mask)
        max_len = max(max_len, len(full_ids))

    # Pad to same length
    for i in range(len(all_ids)):
        pad_len = max_len - len(all_ids[i])
        all_ids[i] = all_ids[i] + [tokenizer.pad_token_id or 0] * pad_len
        all_resp_masks[i] = all_resp_masks[i] + [0] * pad_len

    input_ids = torch.tensor(all_ids, device=device)
    response_mask = torch.tensor(all_resp_masks, dtype=torch.float, device=device)
    seq_len = input_ids.shape[1]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  response_mask: {response_mask.shape}")
    print(f"  prompt_len: {prompt_len}, max_seq_len: {seq_len}")
    for i in range(min(3, batch_size)):
        resp_len = int(response_mask[i].sum().item())
        print(f"  resp[{i}]: {resp_len} tokens, text=\"{responses[i][:80]}...\"")

    # =========================================================================
    # Step 2: Compute entropy → detect phase boundaries
    # =========================================================================
    print(f"\n[Step 2] Computing entropy + phase boundaries...")

    # Use -log_prob approximation (no logits available in this test)
    # Simulate by running forward to get log_probs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # (batch, seq_len, vocab)

    entropy = compute_token_entropy(logits=logits, response_mask=response_mask)

    active_ent = entropy[response_mask > 0]
    print(f"  entropy stats: mean={active_ent.mean():.4f}, std={active_ent.std():.4f}, "
          f"min={active_ent.min():.4f}, max={active_ent.max():.4f}")

    boundaries_batch = detect_phase_boundaries_pure_entropy(
        entropy=entropy,
        response_mask=response_mask,
        window_size=5,
        percentile=70.0,
        min_phase_len=5,
        max_phases=6,
    )

    for i in range(min(3, batch_size)):
        n_phases = len(boundaries_batch[i])
        print(f"  resp[{i}]: {n_phases} phases, boundaries={boundaries_batch[i]}")

        phases = segment_response_into_phases(
            boundaries=boundaries_batch[i],
            response_length=int(response_mask[i].sum().item()) + prompt_len,
            token_ids=input_ids[i],
            tokenizer=tokenizer,
        )
        for k, phase in enumerate(phases):
            text_preview = phase.text[:60].replace('\n', '\\n')
            print(f"    phase {k}: tok[{phase.start_idx}:{phase.end_idx}] \"{text_preview}\"")

    del outputs, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Partial forward (hook) → hidden states at layer L
    # =========================================================================
    layer_L = n_layers * 3 // 4
    print(f"\n[Step 3] Partial forward at layer L={layer_L}...")

    t3 = time.time()
    with torch.no_grad():
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hs = _partial_forward(model, input_ids[:1], position_ids, layer_L)

    print(f"  hidden_states[0]: {hs.shape}, dtype={hs.dtype} ({time.time()-t3:.2f}s)")
    assert not torch.isnan(hs).any(), "NaN in hidden states!"

    # Verify against full forward
    with torch.no_grad():
        full_out = model(input_ids=input_ids[:1], position_ids=position_ids,
                         output_hidden_states=True, use_cache=False)
        hs_full = full_out.hidden_states[layer_L]

    diff = (hs.float() - hs_full.float()).abs()
    max_diff = diff.max().item()
    print(f"  verification vs full forward: max_diff={max_diff:.6e}", end="")
    if max_diff < 1e-3:
        print(" ✓")
    else:
        print(f" ✗ (expected < 1e-3)")
        return False

    del full_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 4: Reconstruct attention → phase attention matrix
    # =========================================================================
    print(f"\n[Step 4] Reconstruct attention + build phase matrices...")

    phase_rewards_batch = []

    for b in range(batch_size):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        resp_start = active[0].item()
        resp_end = active[-1].item() + 1
        n_phases = len(boundaries_batch[b])

        # Exact match reward for last phase
        r_last = compute_score("math", responses[b], ground_truths[b])

        if n_phases <= 1:
            phase_rewards_batch.append(np.array([r_last]))
            if b < 3:
                print(f"  resp[{b}]: 1 phase, reward={r_last:.2f}")
            continue

        t4 = time.time()

        # Partial forward for this sample
        sub_ids = input_ids[b:b+1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            hs_b = _partial_forward(model, sub_ids, pos_ids, layer_L)

            # Slice to response tokens
            hs_resp = hs_b[:, resp_start:resp_end, :]
            pos_resp = torch.arange(resp_start, resp_end, device=device).unsqueeze(0)

            attn = reconstruct_attention_at_layer(
                model=model, layer_idx=layer_L,
                hidden_state=hs_resp, position_ids=pos_resp,
            )

        # Build phase attention matrix (response-relative boundaries)
        boundaries_rel = [bd - resp_start for bd in boundaries_batch[b]]
        phase_attn = build_phase_attention_matrix(attn, boundaries_rel, resp_end - resp_start)

        # Build influence matrix A
        A = build_influence_matrix_A(phase_attn, norm_mode="row")

        # Solve for rewards
        rewards, residual = solve_phase_rewards(A, r_last=r_last, n_phases=n_phases)
        phase_rewards_batch.append(rewards)

        if b < 3:
            print(f"\n  resp[{b}]: {n_phases} phases, outcome={r_last:.2f} ({time.time()-t4:.2f}s)")
            print(f"    phase_attn ({phase_attn.shape}):")
            print(f"    {np.array2string(phase_attn, precision=4, suppress_small=True)}")
            print(f"    A ({A.shape}, norm=row):")
            print(f"    {np.array2string(A, precision=4, suppress_small=True)}")
            print(f"    rewards: {[f'{v:.4f}' for v in rewards]}")
            print(f"    residual: {residual:.6e}")

        del hs_b, hs_resp, attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # Step 5: Compute advantages
    # =========================================================================
    print(f"\n[Step 5] Computing phase & token advantages...")

    # Fake log_probs (not needed for advantage computation)
    log_probs = torch.zeros_like(response_mask)
    index = torch.zeros(batch_size, dtype=torch.long, device=device)  # all same group

    token_adv, phase_adv, phase_rew_t, phase_mask_t = compute_pure_entropy_advantages(
        log_probs=log_probs,
        response_mask=response_mask,
        index=index,
        boundaries_batch=boundaries_batch,
        phase_rewards_batch=phase_rewards_batch,
    )

    print(f"  token_advantages: {token_adv.shape}")
    print(f"  phase_advantages: {phase_adv.shape}")

    for b in range(min(3, batch_size)):
        n_phases = len(boundaries_batch[b])
        rew_b = phase_rew_t[b][:n_phases].tolist()
        adv_b = phase_adv[b][:n_phases].tolist()
        tok_adv_b = token_adv[b][response_mask[b] > 0]

        print(f"\n  resp[{b}] (outcome={compute_score('math', responses[b], ground_truths[b]):.2f}):")
        print(f"    phase rewards:    {[f'{v:.4f}' for v in rew_b]}")
        print(f"    phase advantages: {[f'{v:.4f}' for v in adv_b]}")
        if tok_adv_b.numel() > 0:
            print(f"    token adv: mean={tok_adv_b.mean():.4f}, std={tok_adv_b.std():.4f}, "
                  f"min={tok_adv_b.min():.4f}, max={tok_adv_b.max():.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - t0

    valid_rew = phase_rew_t[phase_mask_t > 0]
    valid_adv = phase_adv[phase_mask_t > 0]
    tok_adv_all = token_adv[response_mask > 0]

    print(f"\n{'='*70}")
    print(f"PIPELINE TEST SUMMARY ({total_time:.1f}s total)")
    print(f"{'='*70}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Layer L:           {layer_L}")
    print(f"  Avg phases:        {np.mean([len(b) for b in boundaries_batch]):.1f}")
    print(f"  Phase rewards:     mean={valid_rew.mean():.4f}, std={valid_rew.std():.4f}")
    print(f"  Phase advantages:  mean={valid_adv.mean():.4f}, std={valid_adv.std():.4f}")
    print(f"  Token advantages:  mean={tok_adv_all.mean():.4f}, std={tok_adv_all.std():.4f}")

    # Sanity checks
    errors = []
    if torch.isnan(token_adv).any():
        errors.append("NaN in token_advantages")
    if torch.isnan(phase_adv).any():
        errors.append("NaN in phase_advantages")
    if torch.isinf(token_adv).any():
        errors.append("Inf in token_advantages")
    if (token_adv[response_mask == 0].abs() > 1e-6).any():
        errors.append("Non-zero advantage outside response mask")

    if errors:
        print(f"\n  ERRORS: {errors}")
        return False
    else:
        print(f"\n  ALL CHECKS PASSED ✓")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end pipeline test for pure-entropy")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of test samples")
    args = parser.parse_args()

    ok = run_pipeline_test(args.model, args.batch_size)
    sys.exit(0 if ok else 1)
