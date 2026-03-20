"""Smoke tests for entropy-based phase boundary detection."""

import torch
import numpy as np
import pytest

from adpo.adpo_algorithm import (
    compute_token_entropy,
    detect_phase_boundaries_entropy,
    detect_phase_boundaries,
    detect_phase_boundaries_adaptive,
)


# ---------------------------------------------------------------------------
# compute_token_entropy
# ---------------------------------------------------------------------------

class TestComputeTokenEntropy:
    def test_from_logits_uniform(self):
        """Uniform distribution → max entropy = log(vocab_size)."""
        batch, seq, vocab = 2, 10, 100
        logits = torch.zeros(batch, seq, vocab)  # uniform
        mask = torch.ones(batch, seq)

        ent = compute_token_entropy(logits=logits, response_mask=mask)
        expected = np.log(vocab)
        assert ent.shape == (batch, seq)
        assert torch.allclose(ent, torch.tensor(expected).float(), atol=1e-4)

    def test_from_logits_peaked(self):
        """One-hot distribution → entropy ≈ 0."""
        batch, seq, vocab = 1, 5, 50
        logits = torch.full((batch, seq, vocab), -1e9)
        logits[:, :, 0] = 0.0  # all mass on token 0
        mask = torch.ones(batch, seq)

        ent = compute_token_entropy(logits=logits, response_mask=mask)
        assert (ent < 0.01).all(), f"Expected near-zero entropy, got {ent}"

    def test_from_log_probs_proxy(self):
        """Fallback: -log_probs as entropy proxy."""
        log_probs = torch.tensor([[-0.1, -2.0, -0.5, -3.0]])
        mask = torch.ones(1, 4)

        ent = compute_token_entropy(log_probs=log_probs, response_mask=mask)
        expected = torch.tensor([[0.1, 2.0, 0.5, 3.0]])
        assert torch.allclose(ent, expected)

    def test_mask_zeros_non_response(self):
        """Non-response tokens should have 0 entropy."""
        log_probs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

        ent = compute_token_entropy(log_probs=log_probs, response_mask=mask)
        assert ent[0, 0].item() == 0.0
        assert ent[0, 3].item() == 0.0
        assert ent[0, 1].item() > 0.0

    def test_raises_without_inputs(self):
        with pytest.raises(ValueError, match="Either logits or log_probs"):
            compute_token_entropy()

    def test_logits_takes_priority(self):
        """When both logits and log_probs given, logits should be used."""
        batch, seq, vocab = 1, 3, 10
        logits = torch.zeros(batch, seq, vocab)  # uniform → entropy = log(10)
        log_probs = torch.tensor([[-0.1, -0.1, -0.1]])  # would give 0.1
        mask = torch.ones(batch, seq)

        ent = compute_token_entropy(log_probs=log_probs, logits=logits, response_mask=mask)
        expected = np.log(vocab)
        # Should use logits (entropy ≈ 2.3), not log_probs (entropy = 0.1)
        assert ent[0, 0].item() > 2.0


# ---------------------------------------------------------------------------
# detect_phase_boundaries_entropy
# ---------------------------------------------------------------------------

class TestDetectPhaseBoundariesEntropy:
    def test_basic_peak_detection(self):
        """High entropy positions should become boundaries."""
        # Simulate: mostly low entropy, with 2 clear peaks
        entropy = torch.zeros(1, 50)
        entropy[0, :] = 0.5  # baseline
        entropy[0, 0] = 0.5  # start of response
        entropy[0, 20] = 5.0  # peak 1
        entropy[0, 40] = 4.0  # peak 2

        mask = torch.ones(1, 50)

        boundaries = detect_phase_boundaries_entropy(
            entropy, mask, percentile=80, min_phase_len=5, max_phases=10,
        )

        assert len(boundaries) == 1  # 1 batch element
        assert 0 in boundaries[0]   # start always included
        assert 20 in boundaries[0]  # peak 1
        assert 40 in boundaries[0]  # peak 2

    def test_min_phase_len_enforced(self):
        """Peaks too close together: only highest should survive."""
        entropy = torch.zeros(1, 30)
        entropy[0, :] = 0.1
        entropy[0, 10] = 5.0  # peak 1
        entropy[0, 13] = 4.5  # peak 2 — only 3 tokens away

        mask = torch.ones(1, 30)

        boundaries = detect_phase_boundaries_entropy(
            entropy, mask, percentile=50, min_phase_len=10, max_phases=10,
        )

        assert 10 in boundaries[0]
        assert 13 not in boundaries[0]  # too close to 10

    def test_max_phases_respected(self):
        """Should not exceed max_phases."""
        entropy = torch.zeros(1, 200)
        entropy[0, :] = 0.1
        # Place 20 peaks far apart
        for i in range(20):
            entropy[0, i * 10] = 5.0 + i * 0.1

        mask = torch.ones(1, 200)

        boundaries = detect_phase_boundaries_entropy(
            entropy, mask, percentile=50, min_phase_len=5, max_phases=5,
        )

        assert len(boundaries[0]) <= 5

    def test_empty_response(self):
        """Empty response mask → [0] boundary."""
        entropy = torch.zeros(1, 10)
        mask = torch.zeros(1, 10)

        boundaries = detect_phase_boundaries_entropy(entropy, mask)
        assert boundaries == [[0]]

    def test_flat_entropy_no_extra_boundaries(self):
        """Uniform entropy → no peaks above threshold → only [start]."""
        entropy = torch.ones(1, 50)  # flat
        mask = torch.ones(1, 50)

        boundaries = detect_phase_boundaries_entropy(
            entropy, mask, percentile=80, min_phase_len=5,
        )

        # percentile(uniform, 80) = 1.0, candidates need val > 1.0
        # All values are exactly 1.0, so no candidates
        assert boundaries == [[0]]

    def test_batch_processing(self):
        """Multiple batch elements processed correctly."""
        entropy = torch.zeros(3, 40)
        mask = torch.ones(3, 40)

        # Batch 0: peak at 20
        entropy[0, :] = 0.1
        entropy[0, 20] = 5.0

        # Batch 1: peaks at 10, 30
        entropy[1, :] = 0.1
        entropy[1, 10] = 5.0
        entropy[1, 30] = 4.0

        # Batch 2: no peaks (flat)
        entropy[2, :] = 1.0

        boundaries = detect_phase_boundaries_entropy(
            entropy, mask, percentile=80, min_phase_len=5, max_phases=10,
        )

        assert len(boundaries) == 3
        assert 20 in boundaries[0]
        assert 10 in boundaries[1]
        assert 30 in boundaries[1]
        assert boundaries[2] == [0]  # flat → only start


# ---------------------------------------------------------------------------
# detect_phase_boundaries dispatcher with entropy
# ---------------------------------------------------------------------------

class TestDispatcherEntropy:
    def test_entropy_method_dispatched(self):
        """method='entropy' should use entropy tensor."""
        neg_log_probs = torch.ones(1, 30) * 0.5  # flat
        entropy = torch.zeros(1, 30)
        entropy[0, :] = 0.1
        entropy[0, 15] = 5.0  # only peak in entropy, not in neg_log_probs

        mask = torch.ones(1, 30)

        boundaries = detect_phase_boundaries(
            neg_log_probs, mask, method="entropy",
            percentile=80, min_phase_len=5, max_phases=10,
            entropy=entropy,
        )

        assert 15 in boundaries[0]

    def test_entropy_fallback_to_neg_log_probs(self):
        """method='entropy' without entropy tensor → uses neg_log_probs."""
        neg_log_probs = torch.zeros(1, 30)
        neg_log_probs[0, :] = 0.1
        neg_log_probs[0, 15] = 5.0

        mask = torch.ones(1, 30)

        boundaries = detect_phase_boundaries(
            neg_log_probs, mask, method="entropy",
            percentile=80, min_phase_len=5, max_phases=10,
            entropy=None,  # no entropy → fallback
        )

        assert 15 in boundaries[0]

    def test_entropy_vs_adaptive_different_results(self):
        """entropy and adaptive should give different results with different signals."""
        # neg_log_probs has peak at 10, entropy has peak at 25
        neg_log_probs = torch.zeros(1, 40)
        neg_log_probs[0, :] = 0.1
        neg_log_probs[0, 10] = 5.0

        entropy = torch.zeros(1, 40)
        entropy[0, :] = 0.1
        entropy[0, 25] = 5.0

        mask = torch.ones(1, 40)

        b_adaptive = detect_phase_boundaries(
            neg_log_probs, mask, method="adaptive",
            percentile=80, min_phase_len=5, max_phases=10,
        )
        b_entropy = detect_phase_boundaries(
            neg_log_probs, mask, method="entropy",
            percentile=80, min_phase_len=5, max_phases=10,
            entropy=entropy,
        )

        assert 10 in b_adaptive[0]
        assert 10 not in b_entropy[0]
        assert 25 in b_entropy[0]
        assert 25 not in b_adaptive[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
