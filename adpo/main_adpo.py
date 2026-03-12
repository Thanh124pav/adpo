"""
Main entry point for ADPO training with verl.

Registers the ADPO advantage estimator, then delegates to verl's standard
GRPO training pipeline.

Usage:
    python -m adpo.main_adpo algorithm.adpo_beta=1.0 ...
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo import core_algos

from adpo.adpo_algorithm import compute_grpo_outcome_advantage_adpo

logger = logging.getLogger(__name__)


def register_adpo_advantage(beta: float = 1.0, norm_adv_by_std: bool = True):
    """Register ADPO as a custom advantage estimator in verl."""
    _original_grpo = core_algos.compute_grpo_outcome_advantage

    def adpo_compute_advantage(token_level_scores, response_mask, index, **kwargs):
        token_log_probs = kwargs.pop("token_log_probs", None)
        if token_log_probs is None:
            logger.warning("[ADPO] token_log_probs not provided, falling back to GRPO.")
            return _original_grpo(token_level_scores, response_mask, index, **kwargs)
        return compute_grpo_outcome_advantage_adpo(
            token_log_probs=token_log_probs,
            rewards=token_level_scores,
            response_mask=response_mask,
            index=index,
            beta=beta,
            norm_adv_by_std=norm_adv_by_std,
        )

    core_algos.compute_grpo_outcome_advantage = adpo_compute_advantage
    logger.info(f"[ADPO] Registered advantage estimator (beta={beta})")


@hydra.main(config_path="../configs", config_name="adpo_trainer", version_base=None)
def main(config: DictConfig):
    """Launch ADPO training."""
    print(OmegaConf.to_yaml(config))
    adpo_beta = config.algorithm.get("adpo_beta", 1.0)
    norm_adv = config.algorithm.get("norm_adv_by_std_in_grpo", True)
    register_adpo_advantage(beta=adpo_beta, norm_adv_by_std=norm_adv)
    trainer = RayPPOTrainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
