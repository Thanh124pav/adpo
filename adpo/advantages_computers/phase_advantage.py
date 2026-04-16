import torch
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


class PhaseAdvantage:
    def __init__(self, config):
        self.alpha = config.algorithm.get("alpha", 0.5)
        self.decay_gamma = config.algorithm.get("decay_gamma", 0.0)
        self.eps = 1e-10

    def compute(self, phase_rewards, phase_mask, response_mask,
               boundaries_batch, index,
               alpha=None, decay_gamma=None, eps=None, **kwargs) -> torch.Tensor:
        if alpha is None:
            alpha = self.alpha
        if decay_gamma is None:
            decay_gamma = self.decay_gamma
        if eps is None:
            eps = self.eps 
        batch_size, seq_len = response_mask.shape
        phase_advs = compute_phase_advantages(phase_rewards, phase_mask, index, alpha, eps)
        phase_ids = build_phase_mask(boundaries_batch, seq_len ,response_mask)
        token_adv = assign_phase_advantages_to_tokens(phase_advs, phase_ids, response_mask, decay_gamma, boundaries_batch)
        for b in range(batch_size):
            n_phases = len(boundaries_batch[b])
            if n_phases > 0:
                token_adv[b] /= n_phases
        return token_adv

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

def compute_phase_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    alpha: float = 0.5,
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

    # Phase advantage = local + global (additive, not weighted)
    phase_advantages = (alpha * local_adv + (1-alpha) * global_adv) * phase_mask

    return phase_advantages

def compute_local_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LOCAL advantages: phase k vs other phases within the SAME response.

    A_{i,k}^local = (r_{i,k} - mean(r_{i,*})) / (std(r_{i,*}) + eps)

    Returns:
        local_advantages: (batch, max_K)
        local_stds: (batch,) std of phase rewards within each response.
    """
    batch_size, max_K = phase_rewards.shape

    phase_count = phase_mask.sum(dim=1).clamp(min=1)
    mean_per_response = (phase_rewards * phase_mask).sum(dim=1) / phase_count

    sq_diff = ((phase_rewards - mean_per_response.unsqueeze(1)) ** 2) * phase_mask
    variance = sq_diff.sum(dim=1) / phase_count.clamp(min=1)
    local_stds = torch.sqrt(variance + eps)

    # Normalize by std
    local_advantages = ((phase_rewards - mean_per_response.unsqueeze(1))
                        / (local_stds.unsqueeze(1) + eps)) * phase_mask

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
        A_{i,k}^global = (score_i - mean_j(score_j)) / (std_j(score_j) + eps)

    Returns:
        global_advantages: (batch, max_K) -- same value per phase within a response.
        global_std: (scalar per group) std of response scores across G generations.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    phase_count = phase_mask.sum(dim=1).clamp(min=1)
    response_scores = (phase_rewards * phase_mask).sum(dim=1) / phase_count

    global_advantages = torch.zeros_like(phase_rewards)
    global_stds = torch.zeros(batch_size, device=device)

    unique_indices = torch.unique(index)

    for uid in unique_indices:
        group_mask = (index == uid)
        group_scores = response_scores[group_mask]
        group_mean = group_scores.mean()
        group_std = group_scores.std() if group_scores.numel() > 1 else torch.tensor(0.0, device=device)

        # Normalize by std
        adv = (group_scores - group_mean) / (group_std + eps)
        global_advantages[group_mask] = adv.unsqueeze(1) * phase_mask[group_mask]
        global_stds[group_mask] = group_std

    return global_advantages, global_stds



def assign_phase_advantages_to_tokens(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
    decay_gamma: float = 0.0,
    boundaries_batch: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Map phase-level advantages back to token-level.

    Hard: A(t) = A^(k(t))
    Decreasing (decay_gamma > 0): Within each phase, token advantages decay
        geometrically so earlier tokens get more credit:
            c = A_phase / (2 * T)
            a'_1 = A * (1 - gamma) * T / (9 * gamma)
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
        c       = A / 3                  (constant baseline per token)
        a'_1    = A / 2                  (initial decay value)
        a_i     = c + a'_i              (token i gets baseline + decayed part)
        a'_{i+1}= a'_i * gamma          (geometric decay)
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device
    token_advantages = torch.zeros(batch_size, seq_len, device=device)

    logged = False
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
            c = A / 3.0
            a_prime = A / 2.0  # a'_1
            for i in range(T):
                token_advantages[b, start + i] = c + a_prime
                a_prime *= gamma

            # Log first response's decay info
            if not logged and T > 1:
                first_adv = token_advantages[b, start].item()
                last_adv = token_advantages[b, end - 1].item()
                print(
                    f"[ADPO Decay] b={b} phase={k} | T={T} A={A:.6e} "
                    f"gamma={gamma} | adv_first={first_adv:.6e} "
                    f"adv_last={last_adv:.6e} diff={first_adv - last_adv:.6e}",
                    flush=True,
                )
        if not logged:
            logged = True

    return token_advantages