import torch
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .base import BaseReward
from adpo.reward_computers.llm_judge import create_judge, PhaseJudge
from adpo.reward_functions import compute_score
logger = logging.getLogger(__name__)

class JudgeReward(BaseReward):
    def __init__(self, config):
        super(self, JudgeReward).__init__()
        algo = config.algorithm
        self.correct_reward_floor = getattr(algo, "correct_reward_floor", 0.5)
        self.incorrect_penalty = getattr(algo, "incorrect_penalty", 0.3)
        self.no_answer_correct_scale = getattr(algo, "no_answer_correct_scale", 0.5)
        self.no_answer_incorrect_scale = getattr(algo, "no_answer_incorrect_scale", 0.1)
        self.overlong_buffer_len = getattr(algo, "overlong_buffer_len", 0)
        self.overlong_penalty_factor = getattr(algo, "overlong_penalty_factor", 1.0)
        self.output_correct_reward = getattr(algo, "output_correct_reward", 0.5)
        self.output_incorrect_reward = getattr(algo, "output_incorrect_reward", 0.0)
        # Judge
        judge_type = getattr(algo, "judge_type", "rule")
        judge_model = getattr(algo, "judge_model", "Qwen/Qwen2.5-7B-Instruct")
        judge_endpoint = getattr(algo, "judge_endpoint", "")
        judge_kwargs = {}
        if judge_type == "endpoint" and judge_endpoint:
            judge_kwargs["endpoint"] = judge_endpoint
        self.judge = create_judge(
            judge_type=judge_type, judge_model=judge_model, **judge_kwargs
        )

        logger.info(
            f"ADPO Trainer: method={self.phase_method}, judge={judge_type}"
        )

    def compute(self, boundaries_batch, response_mask, index, phase_texts_batch, questions, golden_answers, data_sources, full_responses):
        batch_size, seq_len = response_mask.shape
        max_K = max(len(b) for b in boundaries_batch)
        # Step 4: Score phases with judge (with golden answer + ref solutions)
        phase_rewards_list = self.judge.score_phases(
            questions=questions,
            phase_texts=phase_texts_batch,
            golden_answers=golden_answers,
            data_sources=data_sources,
        )

        # Step 5: Compute outcome rewards
        outcome_rewards = []
        for b in range(batch_size):
            if golden_answers[b]:
                r = compute_score(
                    data_source=data_sources[b],
                    solution_str=full_responses[b],
                    ground_truth=golden_answers[b],
                )
            else:
                r = 0.0
            outcome_rewards.append(r)

        # Step 5b: Overlong reward shaping (DAPO-style soft penalty)
        overlong_rewards = [0.0] * batch_size
        if self.overlong_buffer_len > 0:
            max_resp_len = response_mask.shape[1]
            expected_len = max_resp_len - self.overlong_buffer_len
            for b in range(batch_size):
                resp_len = int(response_mask[b].sum().item())
                exceed_len = resp_len - expected_len
                penalty = min(-exceed_len / self.overlong_buffer_len * self.overlong_penalty_factor, 0.0)
                overlong_rewards[b] = penalty

        # Step 6: Build phase reward tensor
        device = response_mask.device
        phase_rewards = torch.zeros(batch_size, max_K, device=device)
        phase_mask_tensor = torch.zeros(batch_size, max_K, device=device)

        for b in range(batch_size):
            n_phases = len(phase_rewards_list[b])
            for k in range(n_phases):
                phase_rewards[b, k] = phase_rewards_list[b][k]
                phase_mask_tensor[b, k] = 1.0

        # Step 6b: Score mapping — 3-tier scaling based on outcome + golden answer
        outcome_tensor = torch.tensor(outcome_rewards, device=device)
        has_golden = torch.tensor([bool(ga) for ga in golden_answers], dtype=torch.bool, device=device)
        no_golden = ~has_golden

        # Response-level mean judge score (from original phase rewards, before scaling)
        phase_count = phase_mask_tensor.sum(dim=1).clamp(min=1)
        response_judge_scores = (phase_rewards * phase_mask_tensor).sum(dim=1) / phase_count

        # For no-golden-answer: classify as "good"/"bad" by comparing to group mean
        group_mean_scores = torch.zeros(batch_size, device=device)
        for uid in torch.unique(index):
            gmask = (index == uid)
            group_mean_scores[gmask] = response_judge_scores[gmask].mean()

        # Build score_scale per response
        score_scale = torch.ones(batch_size, device=device)
        # Tier 1: has golden + correct → 1.0 (unchanged)
        # Tier 2: has golden + incorrect → incorrect_penalty
        score_scale[has_golden & (outcome_tensor < 1.0)] = self.incorrect_penalty
        # Tier 3: no golden + judge-good → no_answer_correct_scale
        score_scale[no_golden & (response_judge_scores >= group_mean_scores)] = self.no_answer_correct_scale
        # Tier 4: no golden + judge-bad → no_answer_incorrect_scale
        score_scale[no_golden & (response_judge_scores < group_mean_scores)] = self.no_answer_incorrect_scale

        phase_rewards = phase_rewards * score_scale.unsqueeze(1)

        # Step 6c: Floor phase rewards for correct responses
        correct_mask = has_golden & (outcome_tensor >= 1.0)
        if correct_mask.any():
            floor_mask = correct_mask.unsqueeze(1) & (phase_mask_tensor > 0)
            phase_rewards = torch.where(
                floor_mask,
                torch.clamp(phase_rewards, min=self.correct_reward_floor),
                phase_rewards,
            )

        # Step 6d: Add overlong penalty to all phase rewards
        if self.overlong_buffer_len > 0:
            overlong_tensor = torch.tensor(overlong_rewards, device=device)
            phase_rewards = phase_rewards + (overlong_tensor.unsqueeze(1) * phase_mask_tensor)

        return phase_rewards