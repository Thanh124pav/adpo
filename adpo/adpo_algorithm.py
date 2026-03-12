"""
ADPO: Advantage Decomposition by Phase-Level Log Probability

Core idea:
1. Segment a generated response into K phases using log-probability
   boundaries (tokens where -log pi spikes = reasoning step transitions).
2. Score each phase independently with LLM-as-Judge.
3. Compute per-phase advantages (GRPO-style normalization within groups).
4. Assign each token the advantage of its parent phase.

This replaces GRPO's uniform response-level advantage with fine-grained
phase-level credit assignment.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhaseSegment:
    """A contiguous segment (phase) of a response."""
    phase_id: int
    start_idx: int      # inclusive
    end_idx: int         # exclusive
    text: str = ""
    reward: float = 0.0
    advantage: float = 0.0


# ---------------------------------------------------------------------------
# Phase Boundary Detection
# ---------------------------------------------------------------------------

def compute_neg_log_probs(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute -log pi masked to response tokens."""
    return (-log_probs) * response_mask


def detect_phase_boundaries_threshold(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float = 2.0,
    min_phase_len: int = 10,
) -> List[List[int]]:
    """Detect phase boundaries using fixed threshold on -log pi.

    boundary(t) = 1[-log pi(a_t | s_t) > delta]
    with minimum spacing of min_phase_len tokens.
    """
    batch_size = neg_log_probs.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        nlp = neg_log_probs[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        boundaries = [start]
        last_boundary = start
        for t in range(start + min_phase_len, end):
            if nlp[t].item() > delta and (t - last_boundary) >= min_phase_len:
                boundaries.append(t)
                last_boundary = t

        all_boundaries.append(boundaries)

    return all_boundaries


def detect_phase_boundaries_adaptive(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 85.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Detect phase boundaries using adaptive per-response percentile.

    Uses the p-th percentile of -log pi as threshold, then selects
    local maxima greedily (highest peaks first, respecting min spacing).
    """
    batch_size = neg_log_probs.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        nlp = neg_log_probs[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        active_nlp = nlp[active].cpu().numpy()
        delta = np.percentile(active_nlp, percentile)

        # Find local maxima above threshold
        candidates = []
        for t in range(start + 1, end - 1):
            val = nlp[t].item()
            if (val > delta
                    and val >= nlp[t - 1].item()
                    and val >= nlp[t + 1].item()):
                candidates.append((t, val))

        # Greedy selection: highest peaks first, respecting min distance
        candidates.sort(key=lambda x: -x[1])
        boundaries = [start]
        for t, val in candidates:
            if len(boundaries) >= max_phases:
                break
            if all(abs(t - b_) >= min_phase_len for b_ in boundaries):
                boundaries.append(t)

        boundaries.sort()
        all_boundaries.append(boundaries)

    return all_boundaries


def detect_phase_boundaries(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    method: str = "adaptive",
    delta: float = 2.0,
    percentile: float = 85.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Unified boundary detection dispatcher."""
    if method == "threshold":
        return detect_phase_boundaries_threshold(
            neg_log_probs, response_mask, delta=delta,
            min_phase_len=min_phase_len,
        )
    elif method == "adaptive":
        return detect_phase_boundaries_adaptive(
            neg_log_probs, response_mask, percentile=percentile,
            min_phase_len=min_phase_len, max_phases=max_phases,
        )
    else:
        raise ValueError(f"Unknown boundary method: {method}")


# ---------------------------------------------------------------------------
# Phase Segmentation
# ---------------------------------------------------------------------------

def segment_response_into_phases(
    boundaries: List[int],
    response_length: int,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[PhaseSegment]:
    """Convert boundary indices into PhaseSegment objects."""
    phases = []
    for k in range(len(boundaries)):
        start = boundaries[k]
        end = boundaries[k + 1] if k + 1 < len(boundaries) else response_length

        text = ""
        if token_ids is not None and tokenizer is not None:
            text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)

        phases.append(PhaseSegment(
            phase_id=k, start_idx=start, end_idx=end, text=text,
        ))
    return phases


def build_phase_mask(
    boundaries_batch: List[List[int]],
    seq_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build tensor mapping each token to its phase index.

    Returns:
        phase_ids: (batch, seq_len) integer tensor.
                   -1 for non-response tokens.
    """
    batch_size = response_mask.shape[0]
    device = response_mask.device
    phase_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)

    for b in range(batch_size):
        boundaries = boundaries_batch[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            continue
        resp_end = active[-1].item() + 1

        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else resp_end
            phase_ids[b, start:end] = k

    return phase_ids


# ---------------------------------------------------------------------------
# Soft Phase Assignment (Generalized Form, Section 5 of theory)
# ---------------------------------------------------------------------------

def compute_soft_phase_weights(
    phase_ids: torch.Tensor,
    boundaries_batch: List[List[int]],
    response_mask: torch.Tensor,
    sigma: float = 0.0,
) -> torch.Tensor:
    """Compute soft phase assignment weights using Gaussian kernel.

    w_{t,k} = exp(-(t - c_k)^2 / 2*sigma^2) / sum_k' exp(...)

    sigma=0: hard assignment (one-hot).
    sigma>0: blended near boundaries.
    sigma->inf: uniform (recovers GRPO).

    Returns:
        weights: (batch, seq_len, max_K) soft assignment weights.
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device
    max_K = max(len(b) for b in boundaries_batch) if boundaries_batch else 1

    if sigma <= 0:
        weights = torch.zeros(batch_size, seq_len, max_K, device=device)
        for b in range(batch_size):
            for t in range(seq_len):
                k = phase_ids[b, t].item()
                if k >= 0:
                    weights[b, t, k] = 1.0
        return weights

    weights = torch.zeros(batch_size, seq_len, max_K, device=device)
    for b in range(batch_size):
        boundaries = boundaries_batch[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            continue
        resp_end = active[-1].item() + 1

        centroids = []
        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else resp_end
            centroids.append((start + end) / 2.0)

        for t in active.tolist():
            log_w = torch.tensor(
                [-((t - c) ** 2) / (2 * sigma ** 2) for c in centroids],
                device=device,
            )
            w = torch.softmax(log_w, dim=0)
            weights[b, t, :len(centroids)] = w

    return weights


# ---------------------------------------------------------------------------
# Phase-Level Advantage Computation
# ---------------------------------------------------------------------------

def compute_phase_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    norm_by_std: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute GRPO-style advantages at the phase level.

    A_i^(k) = r_k^(i) - mean_j(r_k^(j))  [/ std if norm_by_std]

    Compares reward of phase k in response i with average across
    all G generations of the same prompt.
    """
    batch_size, max_K = phase_rewards.shape
    phase_advantages = torch.zeros_like(phase_rewards)
    unique_indices = torch.unique(index)

    for idx in unique_indices:
        group_mask = (index == idx)

        for k in range(max_K):
            k_mask = group_mask & (phase_mask[:, k] > 0)
            if k_mask.sum() <= 1:
                continue

            rewards_k = phase_rewards[k_mask, k]
            mean_k = rewards_k.mean()
            std_k = rewards_k.std()

            if norm_by_std and std_k > eps:
                phase_advantages[k_mask, k] = (rewards_k - mean_k) / (std_k + eps)
            else:
                phase_advantages[k_mask, k] = rewards_k - mean_k

    return phase_advantages


def assign_phase_advantages_to_tokens(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
    soft_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map phase-level advantages back to token-level.

    Hard: A(t) = A^(k(t))
    Soft: A(t) = sum_k w_{t,k} * A^(k)
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device

    if soft_weights is not None:
        # (batch, seq_len, max_K) @ (batch, max_K) -> (batch, seq_len)
        token_advantages = torch.einsum(
            "btk,bk->bt", soft_weights, phase_advantages
        )
    else:
        token_advantages = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            for t in range(seq_len):
                k = phase_ids[b, t].item()
                if k >= 0:
                    token_advantages[b, t] = phase_advantages[b, k]

    return token_advantages * response_mask


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def compute_adpo_phase_advantages(
    log_probs: torch.Tensor,
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    boundaries_batch: List[List[int]],
    norm_by_std: bool = True,
    sigma: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Full ADPO pipeline: phase rewards -> phase advantages -> token advantages.

    Called after LLM-as-Judge has scored each phase.
    """
    seq_len = response_mask.shape[1]

    phase_advantages = compute_phase_advantages(
        phase_rewards=phase_rewards,
        phase_mask=phase_mask,
        index=index,
        norm_by_std=norm_by_std,
        eps=eps,
    )

    phase_ids = build_phase_mask(boundaries_batch, seq_len, response_mask)

    soft_weights = None
    if sigma > 0:
        soft_weights = compute_soft_phase_weights(
            phase_ids, boundaries_batch, response_mask, sigma=sigma,
        )

    token_advantages = assign_phase_advantages_to_tokens(
        phase_advantages=phase_advantages,
        phase_ids=phase_ids,
        response_mask=response_mask,
        soft_weights=soft_weights,
    )

    return token_advantages
