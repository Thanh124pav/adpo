"""
Main entry point for ADPO training with verl.

Registers the ADPO phase-based advantage decomposition, then delegates
to verl's standard GRPO training pipeline.

Usage:
    python -m adpo.main_adpo algorithm.judge_type=endpoint ...
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from verl.trainer.main_ppo import run_ppo

from adpo.adpo_trainer import patch_verl_grpo_with_adpo

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="adpo_trainer", version_base=None)
def main(config: DictConfig):
    """Launch ADPO training."""
    print(OmegaConf.to_yaml(config))

    algo = config.algorithm

    # Patch verl with ADPO phase decomposition + SolutionBank
    patch_verl_grpo_with_adpo(
        judge_type=algo.get("judge_type", "rule"),
        judge_model=algo.get("judge_model", "Qwen/Qwen2.5-7B-Instruct"),
        phase_method=algo.get("phase_method", "adaptive"),
        phase_percentile=algo.get("phase_percentile", 85.0),
        phase_min_len=algo.get("phase_min_len", 10),
        phase_max_K=algo.get("phase_max_K", 10),
        phase_sigma=algo.get("phase_sigma", 0.0),
        max_solutions_per_question=algo.get("max_solutions_per_question", 8),
        solution_bank_dir=algo.get("solution_bank_dir", "data/solutions"),
        max_ref_solutions_in_prompt=algo.get("max_ref_solutions_in_prompt", 3),
    )

    # Launch verl trainer (handles tokenizer, workers, resource pool setup)
    run_ppo(config)


if __name__ == "__main__":
    main()
