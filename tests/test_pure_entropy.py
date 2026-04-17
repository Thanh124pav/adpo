"""
Quick integration test for pure-entropy algorithm.

Tests each component independently, then runs the full pipeline
with a real model to catch runtime errors early.

Usage:
    # Unit tests only (no GPU, no model download):
    python tests/test_pure_entropy.py --unit

    # Full integration test with a real model:
    python tests/test_pure_entropy.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

    # Full test with local model path:
    python tests/test_pure_entropy.py --model /path/to/model
"""

import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, ".")


# ============================================================================
# Unit tests (no GPU, no model needed)
# ============================================================================

def test_phase_boundaries():
    """Test entropy-based phase boundary detection."""
    from adpo.pure_entropy_algorithm import detect_phase_boundaries_pure_entropy

    batch_size, seq_len = 2, 100
    # Create fake entropy: mostly low, with some high-entropy regions
    entropy = torch.zeros(batch_size, seq_len)
    entropy[0, 20:35] = 5.0   # high entropy region
    entropy[0, 60:75] = 4.0   # another high region
    entropy[1, 40:55] = 6.0   # high in second sample

    response_mask = torch.ones(batch_size, seq_len)

    boundaries = detect_phase_boundaries_pure_entropy(
        entropy=entropy,
        response_mask=response_mask,
        window_size=10,
        percentile=75.0,
        min_phase_len=10,
        max_phases=5,
    )

    print(f"  boundaries[0] = {boundaries[0]}")
    print(f"  boundaries[1] = {boundaries[1]}")
    assert len(boundaries) == batch_size
    assert boundaries[0][0] == 0, "First boundary should be position 0"
    assert len(boundaries[0]) >= 2, "Should find at least 2 phases"
    print("  PASSED")


def test_phase_attention_matrix():
    """Test building phase attention matrix from token-level attention."""
    from adpo.pure_entropy_algorithm import build_phase_attention_matrix

    num_heads, seq_len = 4, 50
    # Create fake causal attention weights
    attn = torch.zeros(num_heads, seq_len, seq_len)
    for h in range(num_heads):
        for i in range(seq_len):
            # Uniform attention over previous tokens
            if i > 0:
                attn[h, i, :i] = 1.0 / i

    boundaries = [0, 20, 35]
    resp_end = 50

    phase_attn = build_phase_attention_matrix(attn, boundaries, resp_end)
    print(f"  phase_attn shape: {phase_attn.shape}")
    print(f"  phase_attn:\n{np.array2string(phase_attn, precision=6)}")

    assert phase_attn.shape == (3, 3), f"Expected (3,3), got {phase_attn.shape}"
    # Lower triangular (causal)
    assert phase_attn[0, 1] == 0.0, "Should be 0 (causal: phase 0 can't attend to phase 1)"
    assert phase_attn[0, 2] == 0.0
    assert phase_attn[1, 2] == 0.0
    # Diagonal should be non-zero (self-attention)
    assert phase_attn[0, 0] > 0, "Self-attention should be > 0"
    assert phase_attn[1, 1] > 0
    # Cross-phase attention
    assert phase_attn[1, 0] > 0, "Phase 1 should attend to phase 0"
    print("  PASSED")


def test_influence_matrix_A():
    """Test building influence matrix A from phase attention."""
    from adpo.pure_entropy_algorithm import build_influence_matrix_A

    # 4 phases -> A is 3x3
    phase_attn = np.array([
        [0.045, 0.0,   0.0,   0.0  ],
        [0.012, 0.039, 0.0,   0.0  ],
        [0.009, 0.015, 0.041, 0.0  ],
        [0.007, 0.011, 0.019, 0.036],
    ])

    for norm_mode in ["none", "row", "col", "matrix"]:
        A = build_influence_matrix_A(phase_attn, norm_mode=norm_mode)
        print(f"  norm_mode={norm_mode}: A =\n{np.array2string(A, precision=6)}")
        assert A.shape == (3, 3), f"Expected (3,3), got {A.shape}"
        # Upper triangle should be 0
        assert A[0, 1] == 0.0
        assert A[0, 2] == 0.0
        assert A[1, 2] == 0.0

    # Check "none" preserves raw values
    A_none = build_influence_matrix_A(phase_attn, norm_mode="none")
    assert abs(A_none[0, 0] - 0.012) < 1e-9, f"A[0][0] should be phase_attn[1][0]=0.012, got {A_none[0, 0]}"
    assert abs(A_none[1, 0] - 0.009) < 1e-9
    assert abs(A_none[1, 1] - 0.015) < 1e-9
    assert abs(A_none[2, 2] - 0.019) < 1e-9

    # Check "row" normalization
    A_row = build_influence_matrix_A(phase_attn, norm_mode="row")
    for i in range(3):
        row_sum = A_row[i, :].sum()
        if row_sum > 0:
            assert abs(row_sum - 1.0) < 1e-9, f"Row {i} should sum to 1, got {row_sum}"

    print("  PASSED")


def test_solve_phase_rewards():
    """Test solving the linear system for phase rewards."""
    from adpo.pure_entropy_algorithm import solve_phase_rewards

    # Simple case: 3 phases, A is 2x2
    A = np.array([
        [0.5, 0.0],
        [0.3, 0.4],
    ])
    r_last = 1.0

    rewards, residual = solve_phase_rewards(A, r_last, n_phases=3)
    print(f"  A = {A}")
    print(f"  rewards = {rewards}")
    print(f"  residual = {residual:.6e}")

    assert len(rewards) == 3
    assert rewards[-1] == r_last, f"Last reward should be {r_last}"

    # Verify: A * r[0:1] ≈ r[1:2]
    lhs = A @ rewards[:2]
    rhs = rewards[1:]
    max_err = np.abs(lhs - rhs).max()
    print(f"  verification: max|A*r[0:1] - r[1:2]| = {max_err:.6e}")
    assert max_err < 1e-6, f"Verification failed: max error = {max_err}"

    # Edge case: 1 phase
    rewards_1, _ = solve_phase_rewards(np.zeros((0, 0)), r_last=0.5, n_phases=1)
    assert len(rewards_1) == 1 and rewards_1[0] == 0.5

    # Edge case: 2 phases
    A2 = np.array([[0.8]])
    rewards_2, _ = solve_phase_rewards(A2, r_last=1.0, n_phases=2)
    print(f"  2 phases: rewards = {rewards_2}")
    assert len(rewards_2) == 2
    assert rewards_2[-1] == 1.0

    # Variant: fix the first reward to 0.5 and solve the shifted system.
    rewards_fixed, residual_fixed = solve_phase_rewards(
        A,
        r_last=1.0,
        n_phases=3,
        fixed_first_reward=0.5,
    )
    print(f"  fixed-first rewards = {rewards_fixed}")
    print(f"  fixed-first residual = {residual_fixed:.6e}")

    assert len(rewards_fixed) == 3
    assert rewards_fixed[0] == 0.5
    M = np.array([
        [1.0, 0.0],
        [-A[1, 1], 1.0],
    ])
    b = np.array([A[0, 0] * 0.5, -A[1, 0] * 0.5])
    max_err_fixed = np.abs(M @ rewards_fixed[1:] - b).max()
    print(f"  verification (fixed-first): max|M*y - b| = {max_err_fixed:.6e}")
    assert max_err_fixed < 1e-6, f"Fixed-first verification failed: max error = {max_err_fixed}"

    print("  PASSED")


def test_compute_advantages():
    """Test the full advantage computation pipeline."""
    from adpo.pure_entropy_algorithm import compute_pure_entropy_advantages

    batch_size, seq_len = 4, 50
    log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    index = torch.tensor([0, 0, 1, 1])  # 2 groups of 2

    boundaries_batch = [
        [0, 15, 30],
        [0, 20],
        [0, 10, 25, 40],
        [0, 12, 35],
    ]
    phase_rewards_batch = [
        np.array([0.5, 0.8, 1.0]),
        np.array([0.3, 0.0]),
        np.array([0.2, 0.6, 0.9, 1.0]),
        np.array([0.4, 0.7, 0.0]),
    ]

    token_adv, phase_adv, phase_rew, phase_mask = compute_pure_entropy_advantages(
        log_probs=log_probs,
        response_mask=response_mask,
        index=index,
        boundaries_batch=boundaries_batch,
        phase_rewards_batch=phase_rewards_batch,
    )

    print(f"  token_adv shape: {token_adv.shape}")
    print(f"  phase_adv shape: {phase_adv.shape}")
    print(f"  phase_rew[0]: {phase_rew[0].tolist()}")
    print(f"  phase_adv[0]: {[f'{v:.4f}' for v in phase_adv[0].tolist()]}")
    print(f"  phase_mask[0]: {phase_mask[0].tolist()}")

    assert token_adv.shape == (batch_size, seq_len)
    assert phase_adv.shape[0] == batch_size
    # Non-response tokens should have 0 advantage (but here all are response)
    # Phase advantages should have non-zero values for valid phases
    valid_adv = phase_adv[phase_mask > 0]
    print(f"  valid phase_adv: mean={valid_adv.mean():.4f}, std={valid_adv.std():.4f}")

    print("  PASSED")


def test_token_entropy():
    """Test entropy computation."""
    from adpo.pure_entropy_algorithm import compute_token_entropy

    batch_size, seq_len, vocab = 2, 20, 100
    logits = torch.randn(batch_size, seq_len, vocab)
    mask = torch.ones(batch_size, seq_len)

    entropy = compute_token_entropy(logits=logits, response_mask=mask)
    print(f"  entropy shape: {entropy.shape}")
    print(f"  entropy[0][:5]: {entropy[0][:5].tolist()}")
    assert entropy.shape == (batch_size, seq_len)
    assert (entropy >= 0).all(), "Entropy should be non-negative"

    # From log_probs
    log_probs = torch.randn(batch_size, seq_len) - 2.0  # negative log probs
    entropy2 = compute_token_entropy(log_probs=log_probs, response_mask=mask)
    assert entropy2.shape == (batch_size, seq_len)

    print("  PASSED")


def run_unit_tests():
    """Run all unit tests."""
    tests = [
        ("Token entropy computation", test_token_entropy),
        ("Phase boundary detection", test_phase_boundaries),
        ("Phase attention matrix", test_phase_attention_matrix),
        ("Influence matrix A", test_influence_matrix_A),
        ("Solve phase rewards", test_solve_phase_rewards),
        ("Full advantage pipeline", test_compute_advantages),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Unit tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


# ============================================================================
# Integration test (requires GPU + model)
# ============================================================================

def test_partial_forward_and_reconstruct(model_path: str):
    """Test the full forward + attention reconstruction pipeline with a real model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from adpo.pure_entropy_algorithm import (
        _partial_forward,
        reconstruct_attention_at_layer,
        build_phase_attention_matrix,
        build_influence_matrix_A,
        solve_phase_rewards,
    )

    print(f"\n[INTEGRATION] Loading model: {model_path}")
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
    print(f"  Model loaded: n_layers={n_layers}, device={device}")

    # Tokenize a test prompt
    text = "What is 2+3? Let me think step by step. First, 2+3 equals 5. The answer is 5."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"  Input: {seq_len} tokens")

    # Test 1: _partial_forward
    print(f"\n[INTEGRATION] Test _partial_forward")
    target_layer = n_layers * 3 // 4
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        hs = _partial_forward(model, input_ids, position_ids, target_layer)

    print(f"  hidden_states shape: {hs.shape}")
    print(f"  hidden_states dtype: {hs.dtype}")
    assert hs.shape == (1, seq_len, model.config.hidden_size)
    assert not torch.isnan(hs).any(), "Hidden states contain NaN!"
    assert not torch.isinf(hs).any(), "Hidden states contain Inf!"
    print("  _partial_forward PASSED")

    # Verify against full forward
    print(f"\n[INTEGRATION] Verify against full forward (output_hidden_states=True)")
    with torch.no_grad():
        full_out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hs_full = full_out.hidden_states[target_layer]  # (1, seq_len, hidden_dim)

    # Compare
    diff = (hs.float() - hs_full.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  max |partial - full| = {max_diff:.6e}")
    print(f"  mean|partial - full| = {mean_diff:.6e}")
    if max_diff < 1e-3:
        print("  Verification PASSED (max_diff < 1e-3)")
    else:
        print(f"  WARNING: max_diff={max_diff:.6e} > 1e-3")

    del full_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test 2: Reconstruct attention
    print(f"\n[INTEGRATION] Test reconstruct_attention_at_layer (layer={target_layer})")
    with torch.no_grad():
        attn = reconstruct_attention_at_layer(
            model=model,
            layer_idx=target_layer,
            hidden_state=hs,
            position_ids=position_ids,
        )
    print(f"  attn shape: {attn.shape}")  # (num_heads, seq_len, seq_len)
    assert attn.shape[1] == seq_len and attn.shape[2] == seq_len
    assert not torch.isnan(attn).any(), "Attention contains NaN!"
    # Rows should sum to ~1 (softmax)
    row_sums = attn[0].sum(dim=-1)
    print(f"  attn row sums (head 0): min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    print("  reconstruct_attention PASSED")

    # Test 3: Slice to response and build phase matrix
    print(f"\n[INTEGRATION] Test sliced attention + phase matrix")
    resp_start = 0
    resp_end = seq_len
    hs_resp = hs[:, resp_start:resp_end, :]
    pos_resp = torch.arange(resp_start, resp_end, device=device).unsqueeze(0)

    with torch.no_grad():
        attn_resp = reconstruct_attention_at_layer(
            model=model, layer_idx=target_layer,
            hidden_state=hs_resp, position_ids=pos_resp,
        )
    print(f"  sliced attn shape: {attn_resp.shape}")

    boundaries = [0, seq_len // 3, 2 * seq_len // 3]
    phase_attn = build_phase_attention_matrix(attn_resp, boundaries, resp_end - resp_start)
    print(f"  phase_attn ({phase_attn.shape}):\n{np.array2string(phase_attn, precision=6)}")
    assert phase_attn.shape == (3, 3)
    print("  phase_attn PASSED")

    # Test 4: Influence matrix + solve
    print(f"\n[INTEGRATION] Test influence matrix A + solve")
    A = build_influence_matrix_A(phase_attn, norm_mode="row")
    print(f"  A (norm=row):\n{np.array2string(A, precision=6)}")

    rewards, residual = solve_phase_rewards(A, r_last=1.0, n_phases=3)
    print(f"  rewards: {rewards}")
    print(f"  residual: {residual:.6e}")
    assert len(rewards) == 3
    assert rewards[-1] == 1.0
    print("  solve PASSED")

    print(f"\n{'='*60}")
    print("All integration tests PASSED")
    print(f"{'='*60}")
    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pure-entropy algorithm")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path for integration tests")
    args = parser.parse_args()

    if args.unit or args.model is None:
        ok = run_unit_tests()
        if not ok:
            sys.exit(1)

    if args.model:
        try:
            test_partial_forward_and_reconstruct(args.model)
        except Exception as e:
            print(f"\nINTEGRATION TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if not args.unit and args.model is None:
        print("\nTip: run with --model <path> for integration tests with a real model")
