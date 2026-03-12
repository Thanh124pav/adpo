"""
ADPO: Advantage Decomposition by Log Probability

Core algorithm: Decompose response-level advantage into token-level
advantages using log-probability weighting.

    w_t = (-log pi(a_t | s_t)) / sum_{t'} (-log pi(a_{t'} | s_{t'}))
    A_i(t) = w_t * A_i

This replaces GRPO's uniform advantage broadcast with an information-
theoretic weighting that assigns stronger signal to uncertain tokens.
"""

import torch
from typing import Optional


def compute_adpo_token_weights(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-token weights from log probabilities.

    Args:
        log_probs: Log probabilities of shape (batch, seq_len).
                   Values are log pi(a_t | s_t), i.e. negative or zero.
        response_mask: Binary mask (batch, seq_len), 1 for response tokens.
        beta: Temperature parameter.
              b=0 -> uniform (recovers GRPO).
              b=1 -> standard log-prob weighting (default ADPO).
              b->inf -> only most uncertain token gets signal.
        eps: Small constant for numerical stability.

    Returns:
        weights: Normalized weights on the probability simplex,
                 shape (batch, seq_len). Sums to 1 per sequence.
    """
    # neg_log_probs: -log pi(a_t | s_t) >= 0
    neg_log_probs = -log_probs * response_mask

    if beta == 0.0:
        # Uniform weighting -- recover standard GRPO
        token_counts = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        return response_mask / token_counts

    # Apply temperature: (-log pi)^beta
    weighted = neg_log_probs.pow(beta) * response_mask

    # Normalize per sequence: w_t = weighted_t / sum_t' weighted_t'
    seq_sum = weighted.sum(dim=-1, keepdim=True).clamp(min=eps)
    weights = weighted / seq_sum

    return weights * response_mask


def compute_adpo_advantages(
    token_log_probs: torch.Tensor,
    response_level_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Decompose response-level advantages into token-level advantages.

    A_i(t) = w_t * A_i

    where w_t is proportional to (-log pi(a_t | s_t))^beta, normalized per response.

    Args:
        token_log_probs: (batch, seq_len) log probs from current policy.
        response_level_advantages: (batch,) scalar advantage per response.
        response_mask: (batch, seq_len) binary mask for response tokens.
        beta: Weighting temperature.
        eps: Numerical stability constant.

    Returns:
        token_advantages: (batch, seq_len) decomposed per-token advantages.
    """
    weights = compute_adpo_token_weights(
        token_log_probs, response_mask, beta=beta, eps=eps
    )

    # Broadcast response-level advantage and multiply by per-token weight
    token_advantages = weights * response_level_advantages.unsqueeze(-1)

    return token_advantages


def compute_grpo_outcome_advantage_adpo(
    token_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
    norm_adv_by_std: bool = True,
) -> torch.Tensor:
    """Full ADPO advantage computation -- drop-in replacement for GRPO.

    1. Compute response-level scores (sum of token rewards).
    2. Group-normalize to get response-level advantages.
    3. Decompose into token-level advantages via log-prob weighting.

    Args:
        token_log_probs: (batch, seq_len) log probs from current policy.
        rewards: (batch, seq_len) per-token rewards (typically only last
                 token is nonzero for outcome-based reward).
        response_mask: (batch, seq_len) binary mask for response tokens.
        index: (batch,) group index -- responses with the same index
               belong to the same prompt group.
        beta: ADPO weighting temperature.
        eps: Numerical stability constant.
        norm_adv_by_std: If True, normalize by group std (GRPO).
                         If False, only subtract mean (Dr.GRPO).

    Returns:
        token_advantages: (batch, seq_len) decomposed advantages.
    """
    # Step 1: Response-level scores
    scores = (rewards * response_mask).sum(dim=-1)

    # Step 2: Group-normalize
    unique_indices = torch.unique(index)
    response_advantages = torch.zeros_like(scores)

    for idx in unique_indices:
        mask = index == idx
        group_scores = scores[mask]
        group_mean = group_scores.mean()
        group_std = group_scores.std()

        if norm_adv_by_std and group_std > eps:
            response_advantages[mask] = (group_scores - group_mean) / (group_std + eps)
        else:
            response_advantages[mask] = group_scores - group_mean

    # Step 3: Decompose into token-level advantages
    token_advantages = compute_adpo_advantages(
        token_log_probs=token_log_probs,
        response_level_advantages=response_advantages,
        response_mask=response_mask,
        beta=beta,
        eps=eps,
    )

    return token_advantages
