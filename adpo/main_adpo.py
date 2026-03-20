"""
Main entry point for ADPO training with verl.

Registers the ADPO phase-based advantage decomposition, then delegates
to verl's standard GRPO training pipeline.

Usage:
    python -m adpo.main_adpo algorithm.judge_type=endpoint \
        algorithm.judge_endpoint=http://localhost:8000 ...
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import ray
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo

from adpo.adpo_trainer import ADPOTaskRunner

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="adpo_trainer", version_base=None)
def main(config: DictConfig):
    """Launch ADPO training."""
    config = migrate_legacy_reward_impl(config)
    print(OmegaConf.to_yaml(config))

    # Launch verl trainer with ADPOTaskRunner — the patch is applied
    # inside the Ray worker process where compute_advantage actually runs.
    adpo_task_runner_cls = ray.remote(num_cpus=1)(ADPOTaskRunner)
    run_ppo(config, task_runner_class=adpo_task_runner_cls)


if __name__ == "__main__":
    main()
