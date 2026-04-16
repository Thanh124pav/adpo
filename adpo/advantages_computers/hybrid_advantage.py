import torch
from typing import List, Optional

from adpo.advantages_computers.phase_advantage import (
    compute_phase_advantages,
    build_phase_mask,
    assign_phase_advantages_to_tokens,
)


class HybridAdvantage:
    """Hybrid router: GRPO for mixed groups, ADPO phases for all-wrong groups.

    Mixed group (>=1 correct answer):
        A_i = (outcome_i - mean_group) / (std_group + eps)
        → broadcast uniformly to all response tokens

    All-wrong group (every answer incorrect):
        Full phase advantage pipeline: phase rewards → PhaseAdvantage
        (local + global blend with decay)

    This focuses the expensive phase computation only where it helps most
    (all-wrong groups) while using standard GRPO where the signal is clear.
    """

    def __init__(self, config):
        algo = config.algorithm
        self.alpha = getattr(algo, "alpha", 0.5)
        self.decay_gamma = getattr(algo, "decay_gamma", 0.0)
        self.eps = 1e-10
        # outcome_reward >= threshold counts as "correct"
        self.correct_threshold = getattr(algo, "hybrid_correct_threshold", 1.0)

    def compute(
        self,
        phase_rewards: torch.Tensor,
        phase_mask: torch.Tensor,
        response_mask: torch.Tensor,
        boundaries_batch: List[List[int]],
        index: torch.Tensor,
        outcome_rewards: Optional[List[float]] = None,
        alpha: float = None,
        decay_gamma: float = None,
        eps: float = None,
        **kwargs,
    ) -> torch.Tensor:
        if alpha is None:
            alpha = self.alpha
        if decay_gamma is None:
            decay_gamma = self.decay_gamma
        if eps is None:
            eps = self.eps

        batch_size, seq_len = response_mask.shape
        device = response_mask.device

        mixed_mask, all_wrong_mask = self._classify_groups(
            index, outcome_rewards, batch_size, device
        )

        token_advantages = torch.zeros(batch_size, seq_len, device=device)

        # --- GRPO path for mixed groups ---
        if mixed_mask.any() and outcome_rewards is not None:
            grpo_adv = self._compute_grpo_token_advantages(
                outcome_rewards, index, mixed_mask, seq_len, response_mask, eps, device
            )
            token_advantages[mixed_mask] = grpo_adv[mixed_mask]

        # --- ADPO phase path for all-wrong groups ---
        if all_wrong_mask.any():
            phase_adv = compute_phase_advantages(phase_rewards, phase_mask, index, alpha, eps)
            phase_ids = build_phase_mask(boundaries_batch, seq_len, response_mask)
            token_adv_phase = assign_phase_advantages_to_tokens(
                phase_adv, phase_ids, response_mask, decay_gamma, boundaries_batch
            )
            for b in range(batch_size):
                if all_wrong_mask[b]:
                    n_phases = len(boundaries_batch[b])
                    if n_phases > 0:
                        token_adv_phase[b] /= n_phases
            token_advantages[all_wrong_mask] = token_adv_phase[all_wrong_mask]

        return token_advantages

    def _classify_groups(
        self,
        index: torch.Tensor,
        outcome_rewards: Optional[List[float]],
        batch_size: int,
        device: torch.device,
    ):
        """Return (mixed_mask, all_wrong_mask), both shape (batch,) bool tensors."""
        mixed_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        all_wrong_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if outcome_rewards is None:
            all_wrong_mask[:] = True
            return mixed_mask, all_wrong_mask

        outcomes = torch.tensor(outcome_rewards, dtype=torch.float32, device=device)
        for uid in torch.unique(index):
            group_mask = (index == uid)
            group_outcomes = outcomes[group_mask]
            has_correct = (group_outcomes >= self.correct_threshold).any()
            if has_correct:
                mixed_mask[group_mask] = True
            else:
                all_wrong_mask[group_mask] = True

        return mixed_mask, all_wrong_mask

    def _compute_grpo_token_advantages(
        self,
        outcome_rewards: List[float],
        index: torch.Tensor,
        mixed_mask: torch.Tensor,
        seq_len: int,
        response_mask: torch.Tensor,
        eps: float,
        device: torch.device,
    ) -> torch.Tensor:
        """GRPO: normalize outcome_rewards within each group, broadcast to tokens."""
        batch_size = response_mask.shape[0]
        outcomes = torch.tensor(outcome_rewards, dtype=torch.float32, device=device)
        token_adv = torch.zeros(batch_size, seq_len, device=device)

        for uid in torch.unique(index):
            group_idx = (index == uid).nonzero(as_tuple=True)[0]
            group_scores = outcomes[group_idx]
            group_mean = group_scores.mean()
            group_std = (
                group_scores.std()
                if group_scores.numel() > 1
                else torch.tensor(0.0, device=device)
            )
            normalized = (group_scores - group_mean) / (group_std + eps)

            for local_i, global_b in enumerate(group_idx.tolist()):
                if mixed_mask[global_b]:
                    token_adv[global_b] = normalized[local_i] * response_mask[global_b]

        return token_adv
