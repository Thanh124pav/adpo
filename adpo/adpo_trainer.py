"""
ADPO Trainer -- extends verl's RayPPOTrainer with phase decomposition.

Training loop:
1. Rollout: generate G responses per prompt (standard verl)
2. Phase segmentation: detect phase boundaries using log-prob spikes
3. Phase scoring: LLM-as-Judge evaluates each phase
4. Phase advantages: GRPO-style normalization at phase level
5. Token assignment: map phase advantages to tokens
6. Policy update: standard PPO clipped objective (verl)
"""

import torch
import logging
from typing import List, Optional

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo import core_algos

from adpo.adpo_algorithm import (
    compute_neg_log_probs,
    detect_phase_boundaries,
    build_phase_mask,
    segment_response_into_phases,
    compute_phase_advantages,
    assign_phase_advantages_to_tokens,
    compute_soft_phase_weights,
    compute_adpo_phase_advantages,
)
from adpo.llm_judge import create_judge, PhaseJudge

logger = logging.getLogger(__name__)


class ADPOTrainer(RayPPOTrainer):
    """ADPO Trainer with phase-based advantage decomposition.

    Additional config keys (under algorithm):
        phase_method (str): "adaptive" or "threshold". Default "adaptive".
        phase_delta (float): Fixed threshold for boundary detection. Default 2.0.
        phase_percentile (float): Percentile for adaptive detection. Default 85.0.
        phase_min_len (int): Minimum tokens per phase. Default 10.
        phase_max_K (int): Maximum number of phases. Default 10.
        phase_sigma (float): Soft assignment bandwidth. 0=hard. Default 0.0.
        judge_type (str): "vllm", "api", or "rule". Default "rule".
        judge_model (str): Judge model path. Default "Qwen/Qwen2.5-7B-Instruct".
        norm_adv_by_std (bool): GRPO vs Dr.GRPO normalization.
    """

    def __init__(self, config, tokenizer=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        algo = config.algorithm
        self.phase_method = getattr(algo, "phase_method", "adaptive")
        self.phase_delta = getattr(algo, "phase_delta", 2.0)
        self.phase_percentile = getattr(algo, "phase_percentile", 85.0)
        self.phase_min_len = getattr(algo, "phase_min_len", 10)
        self.phase_max_K = getattr(algo, "phase_max_K", 10)
        self.phase_sigma = getattr(algo, "phase_sigma", 0.0)
        self.norm_adv_by_std = getattr(algo, "norm_adv_by_std_in_grpo", True)

        # Initialize judge
        judge_type = getattr(algo, "judge_type", "rule")
        judge_model = getattr(algo, "judge_model", "Qwen/Qwen2.5-7B-Instruct")
        self.judge = create_judge(judge_type=judge_type, judge_model=judge_model)

        self.tokenizer = tokenizer

        logger.info(
            f"ADPO Trainer initialized: method={self.phase_method}, "
            f"judge={judge_type}, sigma={self.phase_sigma}"
        )

    def compute_advantage(self, data):
        """Override: phase-based advantage decomposition.

        Steps:
        1. Compute -log pi for boundary detection
        2. Detect phase boundaries
        3. Extract phase texts
        4. Score phases with LLM-as-Judge
        5. Compute phase-level advantages
        6. Assign to tokens
        """
        log_probs = data.batch["old_log_probs"]
        response_mask = data.batch["response_mask"]
        index = data.batch["uid"]
        input_ids = data.batch.get("input_ids", None)

        batch_size, seq_len = response_mask.shape

        # Step 1: Compute -log pi
        neg_log_probs = compute_neg_log_probs(log_probs, response_mask)

        # Step 2: Detect phase boundaries
        boundaries_batch = detect_phase_boundaries(
            neg_log_probs=neg_log_probs,
            response_mask=response_mask,
            method=self.phase_method,
            delta=self.phase_delta,
            percentile=self.phase_percentile,
            min_phase_len=self.phase_min_len,
            max_phases=self.phase_max_K,
        )

        # Step 3: Extract phase texts for judge
        max_K = max(len(b) for b in boundaries_batch)
        questions = []
        phase_texts_batch = []
        ground_truths = []
        data_sources = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0

            # Get question text
            question = ""
            if hasattr(data.batch, "prompts"):
                question = data.batch["prompts"][b]
            questions.append(question)

            # Get ground truth if available
            if hasattr(data.batch, "ground_truths"):
                ground_truths.append(data.batch["ground_truths"][b])
                data_sources.append(
                    data.batch.get("data_sources", ["math"] * batch_size)[b]
                )

            # Segment into phases and extract text
            phases = segment_response_into_phases(
                boundaries=boundaries_batch[b],
                response_length=resp_end,
                token_ids=input_ids[b] if input_ids is not None else None,
                tokenizer=self.tokenizer,
            )
            phase_texts_batch.append([p.text for p in phases])

        # Step 4: Score phases with judge
        if ground_truths:
            phase_rewards_list = self.judge.score_phases(
                questions=questions,
                phase_texts=phase_texts_batch,
                ground_truths=ground_truths if hasattr(self.judge, 'score_phases') and 'ground_truths' in self.judge.score_phases.__code__.co_varnames else None,
                data_sources=data_sources if hasattr(self.judge, 'score_phases') and 'data_sources' in self.judge.score_phases.__code__.co_varnames else None,
            )
        else:
            phase_rewards_list = self.judge.score_phases(
                questions=questions,
                phase_texts=phase_texts_batch,
            )

        # Step 5: Pad rewards to tensor
        device = response_mask.device
        phase_rewards = torch.zeros(batch_size, max_K, device=device)
        phase_mask = torch.zeros(batch_size, max_K, device=device)

        for b in range(batch_size):
            n_phases = len(phase_rewards_list[b])
            for k in range(n_phases):
                phase_rewards[b, k] = phase_rewards_list[b][k]
                phase_mask[b, k] = 1.0

        # Step 6: Compute phase advantages and assign to tokens
        token_advantages = compute_adpo_phase_advantages(
            log_probs=log_probs,
            phase_rewards=phase_rewards,
            phase_mask=phase_mask,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            norm_by_std=self.norm_adv_by_std,
            sigma=self.phase_sigma,
        )

        data.batch["advantages"] = token_advantages

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_reward = phase_rewards[phase_mask > 0].mean().item() if phase_mask.sum() > 0 else 0
            logger.info(
                f"[ADPO] avg_phases={avg_phases:.1f}, "
                f"avg_phase_reward={avg_reward:.4f}, "
                f"max_K={max_K}"
            )

        return data


import numpy as np


def patch_verl_grpo_with_adpo(
    tokenizer=None,
    judge_type: str = "rule",
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
    phase_method: str = "adaptive",
    phase_percentile: float = 85.0,
    phase_min_len: int = 10,
    phase_max_K: int = 10,
    phase_sigma: float = 0.0,
    norm_by_std: bool = True,
):
    """Monkey-patch verl's GRPO with ADPO phase decomposition.

    Usage:
        from adpo.adpo_trainer import patch_verl_grpo_with_adpo
        patch_verl_grpo_with_adpo(tokenizer=tokenizer, judge_type="vllm")
    """
    judge = create_judge(judge_type=judge_type, judge_model=judge_model)
    original_fn = core_algos.compute_grpo_outcome_advantage

    def adpo_phase_wrapper(
        token_level_scores,
        response_mask,
        index,
        token_log_probs=None,
        **kwargs,
    ):
        if token_log_probs is None:
            logger.warning("[ADPO] No log probs — falling back to GRPO")
            return original_fn(token_level_scores, response_mask, index, **kwargs)

        neg_log_probs = compute_neg_log_probs(token_log_probs, response_mask)

        boundaries_batch = detect_phase_boundaries(
            neg_log_probs=neg_log_probs,
            response_mask=response_mask,
            method=phase_method,
            percentile=phase_percentile,
            min_phase_len=phase_min_len,
            max_phases=phase_max_K,
        )

        batch_size = response_mask.shape[0]
        max_K = max(len(b) for b in boundaries_batch)

        # Use outcome reward as final-phase reward, neutral for others
        scores = (token_level_scores * response_mask).sum(dim=-1)
        device = response_mask.device
        phase_rewards = torch.full((batch_size, max_K), 0.5, device=device)
        phase_mask = torch.zeros(batch_size, max_K, device=device)

        for b in range(batch_size):
            n = len(boundaries_batch[b])
            for k in range(n):
                phase_mask[b, k] = 1.0
            phase_rewards[b, n - 1] = scores[b]  # final phase gets outcome

        return compute_adpo_phase_advantages(
            log_probs=token_log_probs,
            phase_rewards=phase_rewards,
            phase_mask=phase_mask,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            norm_by_std=norm_by_std,
            sigma=phase_sigma,
        )

    core_algos.compute_grpo_outcome_advantage = adpo_phase_wrapper
    logger.info(f"Patched verl GRPO with ADPO phase decomposition")
