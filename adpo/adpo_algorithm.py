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


def compute_token_entropy(
    log_probs: torch.Tensor = None,
    logits: torch.Tensor = None,
    response_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-token entropy from logits or approximate from log_probs.

    If logits are provided, computes exact entropy:
        H(t) = -sum_v p(v|context) * log p(v|context)

    If only log_probs (of the chosen token) are available, uses -log_prob
    as an approximation (higher -log_prob ≈ higher uncertainty).

    Returns:
        entropy: (batch, seq_len) tensor of per-token entropy values.
    """
    if logits is not None:
        # Exact entropy from full vocabulary distribution
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1)  # (batch, seq_len)
    elif log_probs is not None:
        # Approximation: -log p(chosen token) as proxy for entropy
        # Higher -log_prob → model less certain → higher entropy
        entropy = -log_probs
    else:
        raise ValueError("Either logits or log_probs must be provided")

    if response_mask is not None:
        entropy = entropy * response_mask

    return entropy


def detect_phase_boundaries_entropy(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 80.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Detect phase boundaries using per-token entropy (80/20 rule).

    Selects positions where entropy is in the top (100-percentile)% as
    candidate break points (the start of a new phase). For the default
    percentile=80, this means the top 20% highest-entropy positions.

    Greedy selection picks the highest entropy candidates first, respecting
    min_phase_len spacing.

    Args:
        entropy: (batch, seq_len) per-token entropy values.
        response_mask: (batch, seq_len) binary mask for response tokens.
        percentile: Threshold percentile (default 80 → top 20% are candidates).
        min_phase_len: Minimum tokens between consecutive boundaries.
        max_phases: Maximum number of phases per response.

    Returns:
        List of boundary index lists, one per batch element.
    """
    batch_size = entropy.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        ent = entropy[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        # Compute threshold from active tokens
        active_ent = ent[active].cpu().numpy()
        threshold = np.percentile(active_ent, percentile)

        # Find local maxima above threshold (entropy peaks)
        candidates = []
        for t in range(start + 1, end - 1):
            val = ent[t].item()
            if (val > threshold
                    and val >= ent[t - 1].item()
                    and val >= ent[t + 1].item()):
                candidates.append((t, val))

        # Greedy selection: highest entropy peaks first, respecting min distance
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
    entropy: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Unified boundary detection dispatcher.

    Methods:
        threshold - Fixed -log pi threshold.
        adaptive  - Per-response percentile on -log pi (default).
        entropy   - Per-response percentile on token entropy (80/20 rule).
    """
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
    elif method == "entropy":
        if entropy is None:
            # Fallback: use -log_prob as entropy proxy
            entropy = neg_log_probs
        return detect_phase_boundaries_entropy(
            entropy, response_mask, percentile=percentile,
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
        global_std: (scalar per group) std of response scores across G generations.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    # Response-level score = mean of phase rewards
    phase_count = phase_mask.sum(dim=1).clamp(min=1)
    response_scores = (phase_rewards * phase_mask).sum(dim=1) / phase_count  # (batch,)

    global_advantages = torch.zeros_like(phase_rewards)
    # sigma^global: std of response scores across G generations (one scalar per group)
    # We store per-sample so each response knows its group's std
    global_stds = torch.zeros(batch_size, device=device)

    unique_indices = torch.unique(index)

    for uid in unique_indices:
        group_mask = (index == uid)
        group_scores = response_scores[group_mask]
        group_mean = group_scores.mean()
        group_std = group_scores.std() if group_scores.numel() > 1 else torch.tensor(0.0, device=device)

        # A_global = score_i - mean(scores) (Dr.GRPO: no division by std)
        adv = group_scores - group_mean
        # Broadcast same global advantage to all phases
        global_advantages[group_mask] = adv.unsqueeze(1) * phase_mask[group_mask]

        # sigma^global = std of response scores in this group
        global_stds[group_mask] = group_std

    return global_advantages, global_stds


def compute_phase_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ADPO advantages: adaptive weighted mean of local and global.

    A_{i,k} = lambda * A_{i,k}^local + (1 - lambda) * A_{i,k}^global

    where:
        lambda = sigma^global / (sigma^global + sigma_i^local + eps)

        sigma^global = std of response scores across G generations
        sigma_i^local = std of phase rewards within response i

    Intuition:
    - sigma^global HIGH (responses vary a lot across group)
      → lambda HIGH → trust local signal more (global is noisy)
    - sigma_i^local HIGH (phases vary a lot within response)
      → lambda LOW → trust global signal more (local is noisy)

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
    # local_stds: (batch,) -- sigma_i^local per response

    # Global: response i vs other responses (GRPO-style, no std division)
    global_adv, global_stds = compute_global_advantages(
        phase_rewards, phase_mask, index, eps=eps,
    )
    # global_stds: (batch,) -- sigma^global per response (same within a group)

    # Adaptive lambda = sigma^global / (sigma^global + sigma_i^local + eps)
    # Both are (batch,), broadcast to (batch, max_K) via unsqueeze
    lam = global_stds / (global_stds + local_stds + eps)  # (batch,)
    lam = lam.unsqueeze(1).expand_as(phase_rewards)       # (batch, max_K)

    # Adaptive weighted mean
    phase_advantages = lam * local_adv + (1 - lam) * global_adv
    phase_advantages = phase_advantages * phase_mask

    return phase_advantages


def assign_phase_advantages_to_tokens(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
    soft_weights: Optional[torch.Tensor] = None,
    decay_gamma: float = 0.0,
    boundaries_batch: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Map phase-level advantages back to token-level.

    Hard: A(t) = A^(k(t))
    Soft: A(t) = sum_k w_{t,k} * A^(k)
    Decreasing (decay_gamma > 0): Within each phase, token advantages decay
        geometrically so earlier tokens get more credit:
            c = A_phase / (2 * T)
            a'_1 = A_phase (initial decay component)
            a_i = c + a'_i
            a'_{i+1} = a'_i * gamma
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device

    if decay_gamma > 0 and boundaries_batch is not None:
        token_advantages = _assign_with_decay(
            phase_advantages, phase_ids, response_mask,
            boundaries_batch, decay_gamma,
        )
    elif soft_weights is not None:
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


def _assign_with_decay(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
    boundaries_batch: List[List[int]],
    gamma: float,
) -> torch.Tensor:
    """In-phase advantage decreasing: earlier tokens get more credit.

    For each phase with advantage A and T tokens:
        c       = A / (2 * T)       (constant baseline per token)
        a'_1    = A                  (initial decay value)
        a_i     = c + a'_i          (token i gets baseline + decayed part)
        a'_{i+1}= a'_i * gamma      (geometric decay)
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device
    token_advantages = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        boundaries = boundaries_batch[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            continue
        resp_end = active[-1].item() + 1

        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else resp_end
            T = end - start
            if T <= 0:
                continue

            A = phase_advantages[b, k].item()
            c = A / (2.0 * T)
            a_prime = A  # a'_1
            for i in range(T):
                token_advantages[b, start + i] = c + a_prime
                a_prime *= gamma

    return token_advantages


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
    decay_gamma: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Full ADPO pipeline: phase rewards -> phase advantages -> token advantages.

    Called after LLM-as-Judge has scored each phase.

    Advantages are computed as adaptive weighted mean of:
    - Local: phase k vs other phases within same response
    - Global: response i vs other G responses (Dr.GRPO-style, no std division)

    lambda_k = sigma_k^global / (sigma_k^global + sigma_i^local + eps)
    A_{i,k} = lambda_k * A_local + (1-lambda_k) * A_global

    Token-level mapping modes (mutually exclusive, checked in order):
    - decay_gamma > 0: In-phase decreasing — earlier tokens get more credit.
    - sigma > 0: Soft Gaussian blending near phase boundaries.
    - Otherwise: Hard assignment (all tokens in a phase share the same adv).
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
    if decay_gamma <= 0 and sigma > 0:
        soft_weights = compute_soft_phase_weights(
            phase_ids, boundaries_batch, response_mask, sigma=sigma,
        )

    token_advantages = assign_phase_advantages_to_tokens(
        phase_advantages=phase_advantages,
        phase_ids=phase_ids,
        response_mask=response_mask,
        soft_weights=soft_weights,
        decay_gamma=decay_gamma,
        boundaries_batch=boundaries_batch,
    )

    return token_advantages
