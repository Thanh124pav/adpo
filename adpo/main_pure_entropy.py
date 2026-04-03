"""
Main entry point for Pure-Entropy training with verl.

Uses entropy-driven phase decomposition and attention-based reward
propagation instead of LLM-as-Judge scoring.

Usage:
    python -m adpo.main_pure_entropy \
        algorithm.entropy_window_size=10 \
        algorithm.entropy_percentile=75.0 \
        algorithm.attention_layer=24 \
        ...
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import ray
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo

from adpo.pure_entropy_trainer import PureEntropyTaskRunner

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="pure_entropy_trainer", version_base=None)
def main(config: DictConfig):
    """Launch Pure-Entropy training."""
    config = migrate_legacy_reward_impl(config)
    print(OmegaConf.to_yaml(config))

    # PureEntropy needs GPU for HF model forward (hidden states + attention reconstruction).
    # Use fractional GPU so TaskRunner shares GPU with actor/rollout workers
    # (the HF model forward runs between rollout and update, so no contention).
    task_runner_cls = ray.remote(num_cpus=1, num_gpus=0.1)(PureEntropyTaskRunner)
    run_ppo(config, task_runner_class=task_runner_cls)


if __name__ == "__main__":
    main()
