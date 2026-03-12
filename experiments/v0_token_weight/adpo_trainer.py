"""
ADPO Trainer -- extends verl's RayPPOTrainer to use advantage decomposition.

This module monkey-patches verl's GRPO advantage computation with the ADPO
variant, keeping all other training infrastructure (rollout, reward, KL
penalty, clipping) unchanged.
"""

import torch
import logging

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo import core_algos

from adpo.adpo_algorithm import (
    compute_grpo_outcome_advantage_adpo,
    compute_adpo_token_weights,
)

logger = logging.getLogger(__name__)


class ADPOTrainer(RayPPOTrainer):
    """ADPO Trainer that overrides advantage computation in GRPO.

    Inherits the full verl training loop (rollout -> reward -> advantage ->
    policy update) and only replaces the advantage computation step with
    the log-probability decomposition method.

    Additional config keys (under algorithm):
        adpo_beta (float): Weighting temperature. Default 1.0.
        adpo_norm_adv_by_std (bool): Whether to normalize by group std.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.adpo_beta = getattr(config.algorithm, "adpo_beta", 1.0)
        self.adpo_norm_adv_by_std = getattr(
            config.algorithm, "norm_adv_by_std_in_grpo", True
        )
        logger.info(
            f"ADPO Trainer initialized with beta={self.adpo_beta}, "
            f"norm_adv_by_std={self.adpo_norm_adv_by_std}"
        )

    def compute_advantage(self, data):
        """Override advantage computation with ADPO decomposition."""
        token_log_probs = data.batch["old_log_probs"]
        rewards = data.batch["token_level_scores"]
        response_mask = data.batch["response_mask"]
        index = data.batch["uid"]

        token_advantages = compute_grpo_outcome_advantage_adpo(
            token_log_probs=token_log_probs,
            rewards=rewards,
            response_mask=response_mask,
            index=index,
            beta=self.adpo_beta,
            eps=1e-8,
            norm_adv_by_std=self.adpo_norm_adv_by_std,
        )

        data.batch["advantages"] = token_advantages

        # Log diagnostics
        with torch.no_grad():
            weights = compute_adpo_token_weights(
                token_log_probs, response_mask, beta=self.adpo_beta
            )
            weight_entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
            max_weight = (weights * response_mask).max(dim=-1).values.mean()
            logger.info(
                f"[ADPO] weight entropy={weight_entropy:.4f}, "
                f"max_weight={max_weight:.4f}"
            )

        return data


def patch_verl_grpo_with_adpo(beta: float = 1.0, norm_adv_by_std: bool = True):
    """Monkey-patch verl's core_algos to use ADPO advantage computation.

    Usage:
        from adpo.adpo_trainer import patch_verl_grpo_with_adpo
        patch_verl_grpo_with_adpo(beta=1.0)
        # Then run verl's standard training as usual.
    """
    original_fn = core_algos.compute_grpo_outcome_advantage

    def adpo_advantage_wrapper(
        token_level_scores,
        response_mask,
        index,
        token_log_probs=None,
        **kwargs,
    ):
        if token_log_probs is None:
            logger.warning(
                "token_log_probs not available -- falling back to standard GRPO"
            )
            return original_fn(token_level_scores, response_mask, index, **kwargs)

        return compute_grpo_outcome_advantage_adpo(
            token_log_probs=token_log_probs,
            rewards=token_level_scores,
            response_mask=response_mask,
            index=index,
            beta=beta,
            norm_adv_by_std=norm_adv_by_std,
        )

    core_algos.compute_grpo_outcome_advantage = adpo_advantage_wrapper
    logger.info(f"Patched verl GRPO with ADPO (beta={beta})")
