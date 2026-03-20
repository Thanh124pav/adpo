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

from transformers import AutoTokenizer

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo

from adpo.adpo_trainer import patch_verl_grpo_with_adpo

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="adpo_trainer", version_base=None)
def main(config: DictConfig):
    """Launch ADPO training."""
    config = migrate_legacy_reward_impl(config)
    print(OmegaConf.to_yaml(config))

    algo = config.algorithm
    model_path = config.actor_rollout_ref.model.path

    # Load tokenizer so the ADPO patch can decode input_ids into phase texts
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Patch verl with ADPO phase decomposition + LLM-as-Judge + SolutionBank
    patch_verl_grpo_with_adpo(
        tokenizer=tokenizer,
        judge_type=algo.get("judge_type", "rule"),
        judge_model=algo.get("judge_model", "Qwen/Qwen2.5-7B-Instruct"),
        judge_endpoint=algo.get("judge_endpoint", ""),
        phase_method=algo.get("phase_method", "adaptive"),
        phase_percentile=algo.get("phase_percentile", 85.0),
        phase_min_len=algo.get("phase_min_len", 10),
        phase_max_K=algo.get("phase_max_K", 10),
        phase_sigma=algo.get("phase_sigma", 0.0),
        max_solutions_per_question=algo.get("max_solutions_per_question", 8),
        solution_bank_dir=algo.get("solution_bank_dir", "data/solutions"),
        max_ref_solutions_in_prompt=algo.get("max_ref_solutions_in_prompt", 3),
        solution_bank_save_path=algo.get("solution_bank_save_path", "checkpoints/solution_bank.jsonl"),
        solution_bank_save_freq=algo.get("solution_bank_save_freq", 50),
    )

    # Launch verl trainer (handles workers, resource pool setup)
    run_ppo(config)


if __name__ == "__main__":
    main()
