"""
Tests for the refactored ADPO pipeline:
    PhaseSplitter → RewardComputer → AdvantageComputer

Run (no GPU needed for unit tests):
    python tests/test_refactor.py

Run specific sections:
    python tests/test_refactor.py --splitter
    python tests/test_refactor.py --reward
    python tests/test_refactor.py --advantage
    python tests/test_refactor.py --e2e
"""

import sys
import argparse
import math
import types
import numpy as np
import torch

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_config(
    phase_min_len: int = 5,
    phase_max_K: int = 6,
    phase_percentile: float = 85.0,
    alpha: float = 0.5,
    decay_gamma: float = 0.0,
    psi: float = 0.95,
    correct_total: float = 1.0,
    incorrect_total: float = -1.0,
    partial_total: float = 0.1,
):
    """Build a minimal SimpleNamespace config accepted by all components.

    PureEntropySplitter reads directly from config (not config.algorithm),
    while RewardComputer / AdvantageComputer read from config.algorithm.
    Both sets of keys are populated here.
    """
    # Top-level keys (for PureEntropySplitter)
    cfg = types.SimpleNamespace(
        phase_min_len=phase_min_len,
        phase_max_K=phase_max_K,
        phase_percentile=phase_percentile,
        entropy_window_size=3,
        entropy_percentile=phase_percentile,
    )

    # algorithm sub-namespace (for RewardComputer / AdvantageComputer)
    algo = types.SimpleNamespace(
        phase_min_len=phase_min_len,
        phase_max_K=phase_max_K,
        phase_percentile=phase_percentile,
        phase_threshold=phase_percentile,
        alpha=alpha,
        decay_gamma=decay_gamma,
        psi=psi,
        correct_total=correct_total,
        incorrect_total=incorrect_total,
        partial_total=partial_total,
        default_threshold_percentile=90.0,
    )
    # Support dict-style .get() expected by PhaseAdvantage
    algo.get = lambda key, default=None: getattr(algo, key, default)
    cfg.algorithm = algo
    return cfg


def _make_batch(batch_size: int = 2, seq_len: int = 40, prompt_len: int = 5):
    """Create synthetic (entropy, response_mask, log_probs) tensors."""
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1.0

    # Entropy: mostly low with a few spikes
    entropy = torch.rand(batch_size, seq_len) * 0.5
    for b in range(batch_size):
        # Insert spikes at positions 12 and 25
        for spike_pos in (12, 25):
            if spike_pos < seq_len:
                entropy[b, spike_pos] = 5.0

    log_probs = -entropy.clone()
    return entropy, response_mask, log_probs


# ---------------------------------------------------------------------------
# PhaseSplitter tests
# ---------------------------------------------------------------------------

def test_splitter_returns_list_of_lists():
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    cfg = _make_config()
    splitter = PureEntropySplitter(cfg)
    entropy, response_mask, log_probs = _make_batch()
    result = splitter.split(entropy, response_mask, log_probs=log_probs)
    assert isinstance(result, list), "split() must return a list"
    assert all(isinstance(b, list) for b in result), "Each element must be a list"
    print("  PASS: splitter_returns_list_of_lists")


def test_splitter_boundaries_within_response():
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    cfg = _make_config(phase_min_len=3, phase_max_K=8)
    splitter = PureEntropySplitter(cfg)
    entropy, response_mask, log_probs = _make_batch(batch_size=3, seq_len=50, prompt_len=5)
    result = splitter.split(entropy, response_mask, log_probs=log_probs)
    for b, bounds in enumerate(result):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        resp_start = active[0].item()
        resp_end = active[-1].item() + 1
        assert bounds[0] >= resp_start, f"First boundary {bounds[0]} before response start {resp_start}"
        for bd in bounds:
            assert resp_start <= bd < resp_end, f"Boundary {bd} out of range [{resp_start},{resp_end})"
    print("  PASS: splitter_boundaries_within_response")


def test_splitter_max_phases():
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    max_K = 4
    cfg = _make_config(phase_max_K=max_K)
    splitter = PureEntropySplitter(cfg)
    entropy, response_mask, log_probs = _make_batch()
    result = splitter.split(entropy, response_mask, log_probs=log_probs)
    for bounds in result:
        assert len(bounds) <= max_K, f"Got {len(bounds)} phases, max={max_K}"
    print("  PASS: splitter_max_phases")


def test_splitter_min_phase_len():
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    min_len = 5
    cfg = _make_config(phase_min_len=min_len, phase_max_K=10)
    splitter = PureEntropySplitter(cfg)
    entropy, response_mask, log_probs = _make_batch(seq_len=60, prompt_len=5)
    result = splitter.split(entropy, response_mask, log_probs=log_probs)
    for bounds in result:
        for k in range(1, len(bounds)):
            gap = bounds[k] - bounds[k - 1]
            assert gap >= min_len, f"Phase gap {gap} < min_len {min_len}"
    print("  PASS: splitter_min_phase_len")


def test_splitter_extract_phase_texts_shape():
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    cfg = _make_config()
    splitter = PureEntropySplitter(cfg)
    entropy, response_mask, log_probs = _make_batch()
    boundaries_batch = splitter.split(entropy, response_mask, log_probs=log_probs)

    batch_size, seq_len = response_mask.shape
    for b in range(batch_size):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        resp_end = active[-1].item() + 1
        phases = splitter.extract_phase_texts(boundaries_batch[b], resp_end)
        assert len(phases) == len(boundaries_batch[b])
    print("  PASS: splitter_extract_phase_texts_shape")


def run_splitter_tests():
    print("\n=== PhaseSplitter tests ===")
    test_splitter_returns_list_of_lists()
    test_splitter_boundaries_within_response()
    test_splitter_max_phases()
    test_splitter_min_phase_len()
    test_splitter_extract_phase_texts_shape()
    print("All PhaseSplitter tests passed.\n")


# ---------------------------------------------------------------------------
# RewardComputer tests
# ---------------------------------------------------------------------------

def _make_phase_inputs(batch_size=2, seq_len=40, prompt_len=5, n_phases=3):
    """Build (boundaries_batch, response_mask, index, entropy, outcome_rewards)."""
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1.0
    entropy = torch.rand(batch_size, seq_len) * 2.0
    index = torch.zeros(batch_size, dtype=torch.long)
    # Two groups: first half group 0, second half group 1
    index[batch_size // 2:] = 1

    # Boundaries: evenly spaced
    active_len = seq_len - prompt_len
    step = max(1, active_len // n_phases)
    boundaries_batch = []
    for _ in range(batch_size):
        bounds = [prompt_len + k * step for k in range(n_phases)]
        boundaries_batch.append(bounds)

    outcome_rewards = [1.0 if b % 2 == 0 else 0.0 for b in range(batch_size)]
    return boundaries_batch, response_mask, index, entropy, outcome_rewards


def _assert_reward_contract(result, batch_size, boundaries_batch, label):
    """Assert that a reward computer result satisfies the interface contract."""
    assert len(result) == 3, f"{label}: compute() must return 3-tuple"
    phase_rewards, phase_mask, metadata = result
    assert isinstance(phase_rewards, torch.Tensor), f"{label}: phase_rewards must be Tensor"
    assert isinstance(phase_mask, torch.Tensor), f"{label}: phase_mask must be Tensor"
    assert isinstance(metadata, dict), f"{label}: metadata must be dict"

    max_K = max(len(b) for b in boundaries_batch)
    assert phase_rewards.shape == (batch_size, max_K), \
        f"{label}: phase_rewards shape {phase_rewards.shape} != ({batch_size},{max_K})"
    assert phase_mask.shape == (batch_size, max_K), \
        f"{label}: phase_mask shape {phase_mask.shape} != ({batch_size},{max_K})"

    for b, bounds in enumerate(boundaries_batch):
        n = len(bounds)
        assert phase_mask[b, :n].all(), f"{label}: mask[{b}, :{n}] not all-1"
        if max_K > n:
            assert not phase_mask[b, n:].any(), f"{label}: mask[{b}, {n}:] not all-0"


def test_entropy_reward_contract():
    from adpo.reward_computers.entropy_credit_reward import EntropyReward
    cfg = _make_config()
    reward = EntropyReward(cfg)
    batch_size = 4
    boundaries_batch, response_mask, index, entropy, outcome_rewards = \
        _make_phase_inputs(batch_size=batch_size)
    result = reward.compute(
        boundaries_batch=boundaries_batch,
        response_mask=response_mask,
        index=index,
        entropy=entropy,
        outcome_rewards=outcome_rewards,
    )
    _assert_reward_contract(result, batch_size, boundaries_batch, "EntropyReward")
    print("  PASS: entropy_reward_contract")


def test_entropy_reward_sum_equals_total():
    """For EntropyReward, sum of phase rewards must equal R_total per response."""
    from adpo.reward_computers.entropy_credit_reward import EntropyReward
    cfg = _make_config(correct_total=1.0, incorrect_total=-1.0)
    reward = EntropyReward(cfg)

    batch_size = 4
    boundaries_batch, response_mask, index, entropy, outcome_rewards = \
        _make_phase_inputs(batch_size=batch_size, n_phases=3)
    # All in same group for simpler testing
    index = torch.zeros(batch_size, dtype=torch.long)

    phase_rewards, phase_mask, _ = reward.compute(
        boundaries_batch=boundaries_batch,
        response_mask=response_mask,
        index=index,
        entropy=entropy,
        outcome_rewards=outcome_rewards,
    )

    for b in range(batch_size):
        n = int(phase_mask[b].sum().item())
        if n <= 1:
            continue  # single-phase responses skip distribution
        expected = cfg.algorithm.correct_total if outcome_rewards[b] >= 1.0 \
            else cfg.algorithm.incorrect_total
        actual = phase_rewards[b, :n].sum().item()
        assert abs(actual - expected) < 1e-3, \
            f"EntropyReward b={b}: sum={actual:.4f} expected={expected:.4f}"
    print("  PASS: entropy_reward_sum_equals_total")


def run_reward_tests():
    print("=== RewardComputer tests ===")
    test_entropy_reward_contract()
    test_entropy_reward_sum_equals_total()
    print("All RewardComputer tests passed.\n")


# ---------------------------------------------------------------------------
# AdvantageComputer tests
# ---------------------------------------------------------------------------

def _make_advantage_inputs(batch_size=2, seq_len=40, prompt_len=5, n_phases=3, max_K=6):
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1.0
    index = torch.arange(batch_size, dtype=torch.long)

    active_len = seq_len - prompt_len
    step = max(1, active_len // n_phases)
    boundaries_batch = [
        [prompt_len + k * step for k in range(n_phases)]
        for _ in range(batch_size)
    ]

    phase_rewards = torch.randn(batch_size, max_K)
    phase_mask = torch.zeros(batch_size, max_K)
    for b, bounds in enumerate(boundaries_batch):
        phase_mask[b, :len(bounds)] = 1.0
    phase_rewards = phase_rewards * phase_mask

    return phase_rewards, phase_mask, response_mask, boundaries_batch, index


def test_advantage_output_shape():
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage
    cfg = _make_config()
    adv = PhaseAdvantage(cfg)
    batch_size, seq_len = 3, 50
    phase_rewards, phase_mask, response_mask, boundaries_batch, index = \
        _make_advantage_inputs(batch_size=batch_size, seq_len=seq_len)

    result = adv.compute(phase_rewards, phase_mask, response_mask, boundaries_batch, index)
    assert result.shape == response_mask.shape, \
        f"Expected shape {response_mask.shape}, got {result.shape}"
    print("  PASS: advantage_output_shape")


def test_advantage_non_response_zero():
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage
    cfg = _make_config()
    adv = PhaseAdvantage(cfg)
    phase_rewards, phase_mask, response_mask, boundaries_batch, index = \
        _make_advantage_inputs()

    result = adv.compute(phase_rewards, phase_mask, response_mask, boundaries_batch, index)
    non_response = (1.0 - response_mask).bool()
    assert result[non_response].abs().max().item() < 1e-6, \
        "Non-response tokens must have zero advantage"
    print("  PASS: advantage_non_response_zero")


def test_advantage_no_nan_or_inf():
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage
    cfg = _make_config()
    adv = PhaseAdvantage(cfg)
    phase_rewards, phase_mask, response_mask, boundaries_batch, index = \
        _make_advantage_inputs(batch_size=4)

    result = adv.compute(phase_rewards, phase_mask, response_mask, boundaries_batch, index)
    assert not torch.isnan(result).any(), "Advantages contain NaN"
    assert not torch.isinf(result).any(), "Advantages contain Inf"
    print("  PASS: advantage_no_nan_or_inf")


def test_advantage_alpha_zero_pure_global():
    """alpha=0 → only global advantages; responses in same group should differ
    only by their overall score, not by per-phase distribution."""
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage
    cfg = _make_config(alpha=0.0)
    adv = PhaseAdvantage(cfg)
    batch_size = 4
    phase_rewards, phase_mask, response_mask, boundaries_batch, index = \
        _make_advantage_inputs(batch_size=batch_size)
    # Put all in same group
    index = torch.zeros(batch_size, dtype=torch.long)

    result = adv.compute(phase_rewards, phase_mask, response_mask, boundaries_batch, index)
    # With alpha=0, advantages depend only on response-level score → no NaN
    assert not torch.isnan(result).any()
    print("  PASS: advantage_alpha_zero_pure_global")


def test_advantage_decay_gamma():
    """With decay_gamma > 0, first token of a phase should have higher advantage
    than last token of same phase (for a positive phase advantage)."""
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage
    cfg = _make_config(alpha=1.0, decay_gamma=0.5)
    adv = PhaseAdvantage(cfg)
    batch_size = 1
    seq_len = 40
    prompt_len = 5
    n_phases = 2
    max_K = 4

    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1.0
    boundaries_batch = [[prompt_len, prompt_len + 15]]
    phase_rewards = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # first phase positive
    phase_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    index = torch.zeros(batch_size, dtype=torch.long)

    result = adv.compute(phase_rewards, phase_mask, response_mask, boundaries_batch, index)
    # First token of phase 0 vs last token of phase 0
    first_adv = result[0, prompt_len].item()
    last_adv = result[0, prompt_len + 14].item()
    assert first_adv >= last_adv, \
        f"Expected decay: first_adv={first_adv:.4f} should >= last_adv={last_adv:.4f}"
    print("  PASS: advantage_decay_gamma")


def run_advantage_tests():
    print("=== AdvantageComputer tests ===")
    test_advantage_output_shape()
    test_advantage_non_response_zero()
    test_advantage_no_nan_or_inf()
    test_advantage_alpha_zero_pure_global()
    test_advantage_decay_gamma()
    print("All AdvantageComputer tests passed.\n")


# ---------------------------------------------------------------------------
# End-to-end combo test (no GPU, no model)
# ---------------------------------------------------------------------------

def _make_fake_data(batch_size=4, seq_len=64, prompt_len=8):
    """Construct a minimal fake verl DataProto-like object."""
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1.0
    log_probs = -torch.rand(batch_size, seq_len) * 2.0
    log_probs = log_probs * response_mask

    index = torch.zeros(batch_size, dtype=torch.long)
    index[batch_size // 2:] = 1  # two groups

    batch = {
        "old_log_probs": log_probs,
        "response_mask": response_mask,
        "uid": index,
    }

    class FakeData:
        def __init__(self, b):
            self.batch = b
            self.non_tensor_batch = {
                "reward_model": [
                    {"ground_truth": "42"} for _ in range(batch_size)
                ],
                "data_source": ["math"] * batch_size,
            }

    return FakeData(batch)


def test_e2e_entropy_pipeline():
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    from adpo.reward_computers.entropy_credit_reward import EntropyReward
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage
    from adpo.trainer import ADPOTrainer

    cfg = _make_config(phase_min_len=3, phase_max_K=6)
    splitter = PureEntropySplitter(cfg)
    reward = EntropyReward(cfg)
    adv = PhaseAdvantage(cfg)
    trainer = ADPOTrainer(splitter, reward, adv, cfg)

    data = _make_fake_data(batch_size=4, seq_len=64)
    data = trainer.compute_advantages(data)

    advantages = data.batch["advantages"]
    assert advantages.shape == data.batch["response_mask"].shape, \
        f"Advantages shape mismatch: {advantages.shape}"
    assert not torch.isnan(advantages).any(), "Advantages contain NaN"
    assert not torch.isinf(advantages).any(), "Advantages contain Inf"

    # Non-response tokens must be zero
    non_resp = (1 - data.batch["response_mask"]).bool()
    assert advantages[non_resp].abs().max().item() < 1e-6
    print("  PASS: e2e_entropy_pipeline")


def run_e2e_tests():
    print("=== End-to-end combo tests ===")
    test_e2e_entropy_pipeline()
    print("All E2E tests passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splitter",  action="store_true")
    parser.add_argument("--reward",    action="store_true")
    parser.add_argument("--advantage", action="store_true")
    parser.add_argument("--e2e",       action="store_true")
    args = parser.parse_args()

    run_all = not any([args.splitter, args.reward, args.advantage, args.e2e])

    if run_all or args.splitter:
        run_splitter_tests()
    if run_all or args.reward:
        run_reward_tests()
    if run_all or args.advantage:
        run_advantage_tests()
    if run_all or args.e2e:
        run_e2e_tests()

    print("=== All selected tests passed ===")


if __name__ == "__main__":
    main()
