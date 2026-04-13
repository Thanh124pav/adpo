"""
Main entry point for Entropy-Credit training with verl.

Uses entropy-driven phase decomposition and entropy-based credit
assignment. No attention reconstruction needed — much faster than
pure-entropy.

Usage:
    python -m adpo.main_entropy_credit \
        algorithm.psi=0.95 \
        algorithm.default_threshold_percentile=90.0 \
        ...
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import ray
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo

from adpo.entropy_credit_trainer import EntropyCreditTaskRunner

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="entropy_credit_trainer", version_base=None)
def main(config: DictConfig):
    """Launch Entropy-Credit training."""
    config = migrate_legacy_reward_impl(config)
    print(OmegaConf.to_yaml(config))

    # No GPU needed for TaskRunner (no HF model forward pass)
    task_runner_cls = ray.remote(num_cpus=1)(EntropyCreditTaskRunner)
    run_ppo(config, task_runner_class=task_runner_cls)


if __name__ == "__main__":
    main()
