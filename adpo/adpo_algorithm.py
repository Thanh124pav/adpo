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

def compute_local_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LOCAL advantages: phase k vs other phases within the SAME response.

    A_{i,k}^local = r_{i,k} - mean(r_{i,1}, ..., r_{i,K})

    This tells us: "Is phase k better or worse than the average phase
    in this response?"

    Returns:
        local_advantages: (batch, max_K)
        local_stds: (batch,) std of phase rewards within each response.
    """
    batch_size, max_K = phase_rewards.shape

    # Mean reward per response (only over existing phases)
    phase_count = phase_mask.sum(dim=1).clamp(min=1)  # (batch,)
    mean_per_response = (phase_rewards * phase_mask).sum(dim=1) / phase_count  # (batch,)

    # Local advantage: r_{i,k} - mean_k(r_{i,k})
    local_advantages = (phase_rewards - mean_per_response.unsqueeze(1)) * phase_mask

    # Std per response (for adaptive lambda)
    # sigma_i^local = std of phase rewards within response i
    sq_diff = ((phase_rewards - mean_per_response.unsqueeze(1)) ** 2) * phase_mask
    variance = sq_diff.sum(dim=1) / phase_count.clamp(min=1)
    local_stds = torch.sqrt(variance + eps)  # (batch,)

    return local_advantages, local_stds


def compute_global_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GLOBAL advantages: response i vs other G responses (GRPO-style).

    Uses the MEAN of phase rewards as overall response score:
        score_i = mean(r_{i,1}, ..., r_{i,K})
        A_{i,k}^global = score_i - mean_j(score_j)

    This tells us: "Is this response overall better or worse than the
    group average?" Applied uniformly to all phases.

    NOTE: Does NOT divide by std (Dr.GRPO style) to avoid the
    small-std amplification problem.

    Returns:
        global_advantages: (batch, max_K) -- same value per phase within a response.
        global_stds: (max_K,) std of per-phase rewards across group.
                     Indexed by phase k for the adaptive lambda formula.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    # Response-level score = mean of phase rewards
    phase_count = phase_mask.sum(dim=1).clamp(min=1)
    response_scores = (phase_rewards * phase_mask).sum(dim=1) / phase_count  # (batch,)

    global_advantages = torch.zeros_like(phase_rewards)
    # sigma_k^global: std of phase k rewards across all G responses in group
    global_stds = torch.zeros(max_K, device=device)

    unique_indices = torch.unique(index)

    for uid in unique_indices:
        group_mask = (index == uid)
        group_scores = response_scores[group_mask]
        group_mean = group_scores.mean()

        # A_global = score_i - mean(scores) (Dr.GRPO: no division by std)
        adv = group_scores - group_mean
        # Broadcast same global advantage to all phases
        global_advantages[group_mask] = adv.unsqueeze(1) * phase_mask[group_mask]

        # Compute sigma_k^global for each phase k across this group
        for k in range(max_K):
            k_mask = group_mask & (phase_mask[:, k] > 0)
            if k_mask.sum() > 1:
                global_stds[k] = phase_rewards[k_mask, k].std()

    return global_advantages, global_stds


def compute_phase_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ADPO advantages: adaptive weighted mean of local and global.

    A_{i,k} = lambda_k * A_{i,k}^local + (1 - lambda_k) * A_{i,k}^global

    where:
        lambda_k = sigma_k^global / (sigma_k^global + sigma_i^local + eps)

    Intuition:
    - sigma_k^global HIGH (cross-generation variance large for phase k)
      → lambda_k HIGH → trust local signal more (global is noisy)
    - sigma_i^local HIGH (within-response variance large)
      → lambda_k LOW → trust global signal more (local is noisy)
    - Both low → balanced mix

    This avoids Dr.GRPO's problem: instead of dividing by std (which
    amplifies noise when std≈0), we adaptively weight between two
    complementary signals.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    # Local: phase k vs other phases in same response
    local_adv, local_stds = compute_local_advantages(
        phase_rewards, phase_mask, eps=eps,
    )

    # Global: response i vs other responses (GRPO-style, no std division)
    global_adv, global_stds = compute_global_advantages(
        phase_rewards, phase_mask, index, eps=eps,
    )

    # Adaptive lambda_k = sigma_k^global / (sigma_k^global + sigma_i^local + eps)
    # global_stds: (max_K,)  -- per phase across group
    # local_stds:  (batch,)  -- per response across phases
    lambda_k = global_stds.unsqueeze(0) / (
        global_stds.unsqueeze(0) + local_stds.unsqueeze(1) + eps
    )  # (batch, max_K)

    # Adaptive weighted mean
    phase_advantages = lambda_k * local_adv + (1 - lambda_k) * global_adv
    phase_advantages = phase_advantages * phase_mask

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
    sigma: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Full ADPO pipeline: phase rewards -> phase advantages -> token advantages.

    Called after LLM-as-Judge has scored each phase.

    Advantages are computed as adaptive weighted mean of:
    - Local: phase k vs other phases within same response
    - Global: response i vs other G responses (Dr.GRPO-style, no std division)

    lambda_k = sigma_k^global / (sigma_k^global + sigma_i^local + eps)
    A_{i,k} = lambda_k * A_local + (1-lambda_k) * A_global
    """
    seq_len = response_mask.shape[1]

    phase_advantages = compute_phase_advantages(
        phase_rewards=phase_rewards,
        phase_mask=phase_mask,
        index=index,
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
