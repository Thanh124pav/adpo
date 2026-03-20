"""
ADPO Trainer -- extends verl's RayPPOTrainer with phase decomposition.

Training loop:
1. Rollout: generate G responses per prompt (standard verl)
2. Phase segmentation: detect phase boundaries using log-prob spikes
3. Phase scoring: LLM-as-Judge evaluates each phase, using golden answer
   and reference solutions from SolutionBank as context
4. Phase advantages: GRPO-style normalization at phase level
5. Token assignment: map phase advantages to tokens
6. Solution feedback: correct generations are added to SolutionBank
7. Policy update: standard PPO clipped objective (verl)
"""

import torch
import numpy as np
import logging
from typing import List, Optional

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo import core_algos

from adpo.adpo_algorithm import (
    compute_neg_log_probs,
    detect_phase_boundaries,
    build_phase_mask,
    segment_response_into_phases,
    compute_adpo_phase_advantages,
)
from adpo.llm_judge import create_judge, PhaseJudge
from adpo.solution_bank import SolutionBank
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


class ADPOTrainer(RayPPOTrainer):
    """ADPO Trainer with phase-based advantage decomposition + SolutionBank.

    Additional config keys (under algorithm):
        phase_method (str): "adaptive" or "threshold". Default "adaptive".
        phase_delta (float): Fixed threshold. Default 2.0.
        phase_percentile (float): Percentile threshold. Default 85.0.
        phase_min_len (int): Min tokens per phase. Default 10.
        phase_max_K (int): Max phases per response. Default 10.
        phase_sigma (float): Soft assignment bandwidth. Default 0.0.
        judge_type (str): "vllm", "api", or "rule". Default "rule".
        judge_model (str): Judge model. Default "Qwen/Qwen2.5-7B-Instruct".
        max_solutions_per_question (int): SolutionBank capacity. Default 8.
        solution_bank_dir (str): Pre-generated solutions dir. Default "data/solutions".
        solution_bank_save_path (str): Where to save bank. Default "checkpoints/solution_bank.jsonl".
        solution_bank_save_freq (int): Save every N steps. Default 50.
        max_ref_solutions_in_prompt (int): Ref solutions shown to judge. Default 3.
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
        self.max_ref_in_prompt = getattr(algo, "max_ref_solutions_in_prompt", 3)

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

        # SolutionBank
        max_sols = getattr(algo, "max_solutions_per_question", 8)
        self.solution_bank = SolutionBank(max_solutions_per_question=max_sols)

        sol_dir = getattr(algo, "solution_bank_dir", "data/solutions")
        self.solution_bank.load_from_directory(sol_dir)

        self.sol_save_path = getattr(
            algo, "solution_bank_save_path", "checkpoints/solution_bank.jsonl"
        )
        self.sol_save_freq = getattr(algo, "solution_bank_save_freq", 50)
        self._step_count = 0

        self.tokenizer = tokenizer

        logger.info(
            f"ADPO Trainer: method={self.phase_method}, judge={judge_type}, "
            f"solution_bank={self.solution_bank}"
        )

    def compute_advantage(self, data):
        """Phase-based advantage with SolutionBank-enhanced judge."""
        log_probs = data.batch["old_log_probs"]
        response_mask = data.batch["response_mask"]
        index = data.batch["uid"]
        input_ids = data.batch.get("input_ids", None)

        batch_size, seq_len = response_mask.shape

        # Step 1: -log pi
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

        # Step 3: Extract texts, golden answers, reference solutions
        max_K = max(len(b) for b in boundaries_batch)
        questions = []
        phase_texts_batch = []
        golden_answers = []
        ref_solutions_batch = []
        full_responses = []
        data_sources = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0

            # Question
            question = ""
            if hasattr(data.batch, "prompts"):
                question = data.batch["prompts"][b]
            questions.append(question)

            # Golden answer
            gt = ""
            ds = "math"
            if hasattr(data.batch, "ground_truths"):
                gt = data.batch["ground_truths"][b]
            if hasattr(data.batch, "data_sources"):
                ds = data.batch["data_sources"][b]
            golden_answers.append(gt)
            data_sources.append(ds)

            # Reference solutions from SolutionBank
            refs = self.solution_bank.get_solutions(
                question, max_return=self.max_ref_in_prompt
            )
            ref_solutions_batch.append(refs)

            # Segment into phases
            phases = segment_response_into_phases(
                boundaries=boundaries_batch[b],
                response_length=resp_end,
                token_ids=input_ids[b] if input_ids is not None else None,
                tokenizer=self.tokenizer,
            )
            phase_texts_batch.append([p.text for p in phases])

            # Full response for later solution feedback
            full_text = ""
            if input_ids is not None and self.tokenizer is not None:
                resp_start = active[0].item() if len(active) > 0 else 0
                full_text = self.tokenizer.decode(
                    input_ids[b][resp_start:resp_end], skip_special_tokens=True
                )
            full_responses.append(full_text)

        # Step 4: Score phases with judge (with golden answer + ref solutions)
        phase_rewards_list = self.judge.score_phases(
            questions=questions,
            phase_texts=phase_texts_batch,
            golden_answers=golden_answers,
            reference_solutions=ref_solutions_batch,
            data_sources=data_sources,
        )

        # Step 5: Compute outcome rewards and feed correct solutions back
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

        added = self.solution_bank.add_correct_generations(
            questions=questions,
            responses=full_responses,
            rewards=outcome_rewards,
            ground_truths=golden_answers,
            reward_threshold=1.0,
        )
        if added > 0:
            logger.info(f"[ADPO] Added {added} correct solutions to bank")

        # Periodic save
        self._step_count += 1
        if self._step_count % self.sol_save_freq == 0:
            self.solution_bank.save_to_jsonl(self.sol_save_path)

        # Step 6: Build phase reward tensor
        device = response_mask.device
        phase_rewards = torch.zeros(batch_size, max_K, device=device)
        phase_mask_tensor = torch.zeros(batch_size, max_K, device=device)

        for b in range(batch_size):
            n_phases = len(phase_rewards_list[b])
            for k in range(n_phases):
                phase_rewards[b, k] = phase_rewards_list[b][k]
                phase_mask_tensor[b, k] = 1.0

        # Step 7: Phase advantages -> token advantages
        token_advantages = compute_adpo_phase_advantages(
            log_probs=log_probs,
            phase_rewards=phase_rewards,
            phase_mask=phase_mask_tensor,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            sigma=self.phase_sigma,
        )

        data.batch["advantages"] = token_advantages

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_reward = phase_rewards[phase_mask_tensor > 0].mean().item() if phase_mask_tensor.sum() > 0 else 0
            avg_outcome = np.mean(outcome_rewards)
            bank_stats = self.solution_bank.get_stats()
            logger.info(
                f"[ADPO] phases={avg_phases:.1f}, "
                f"phase_reward={avg_reward:.3f}, "
                f"outcome={avg_outcome:.3f}, "
                f"bank={bank_stats['n_solutions']} sols / {bank_stats['n_questions']} qs"
            )

        return data


def patch_verl_grpo_with_adpo(
    tokenizer=None,
    judge_type: str = "rule",
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
    judge_endpoint: str = "",
    phase_method: str = "adaptive",
    phase_percentile: float = 85.0,
    phase_min_len: int = 10,
    phase_max_K: int = 10,
    phase_sigma: float = 0.0,
    max_solutions_per_question: int = 8,
    solution_bank_dir: str = "data/solutions",
    max_ref_solutions_in_prompt: int = 3,
    solution_bank_save_path: str = "checkpoints/solution_bank.jsonl",
    solution_bank_save_freq: int = 50,
):
    """Monkey-patch verl's RayPPOTrainer.compute_advantage with ADPO phase
    decomposition + LLM-as-Judge + SolutionBank.

    This patches at the compute_advantage level (not compute_grpo_outcome_advantage)
    so we have access to the full data.batch, which is required for:
    - Decoding input_ids into phase texts for the LLM judge
    - Reading prompts, ground_truths, data_sources from the batch
    - Feeding correct solutions back into the SolutionBank

    Usage:
        from adpo.adpo_trainer import patch_verl_grpo_with_adpo
        patch_verl_grpo_with_adpo(
            tokenizer=tokenizer,
            judge_type="endpoint",
            judge_endpoint="http://localhost:8000",
        )
    """
    judge_kwargs = {}
    if judge_type == "endpoint" and judge_endpoint:
        judge_kwargs["endpoint"] = judge_endpoint
    judge = create_judge(judge_type=judge_type, judge_model=judge_model, **judge_kwargs)
    solution_bank = SolutionBank(max_solutions_per_question=max_solutions_per_question)
    solution_bank.load_from_directory(solution_bank_dir)

    _step_count = [0]  # mutable counter for closure

    original_compute_advantage = RayPPOTrainer.compute_advantage

    def adpo_compute_advantage(self_trainer, data):
        """ADPO phase-based advantage with LLM-as-Judge scoring."""
        nonlocal tokenizer

        # Lazy-load tokenizer from trainer if not provided
        if tokenizer is None:
            if hasattr(self_trainer, "tokenizer") and self_trainer.tokenizer is not None:
                tokenizer = self_trainer.tokenizer
            else:
                logger.warning(
                    "[ADPO] No tokenizer available — falling back to original compute_advantage"
                )
                return original_compute_advantage(self_trainer, data)

        log_probs = data.batch["old_log_probs"]
        response_mask = data.batch["response_mask"]
        index = data.batch["uid"]
        input_ids = data.batch.get("input_ids", None)

        if input_ids is None:
            logger.warning("[ADPO] No input_ids in batch — falling back to original compute_advantage")
            return original_compute_advantage(self_trainer, data)

        batch_size, seq_len = response_mask.shape

        # Step 1: -log pi
        neg_log_probs = compute_neg_log_probs(log_probs, response_mask)

        # Step 2: Detect phase boundaries
        boundaries_batch = detect_phase_boundaries(
            neg_log_probs=neg_log_probs,
            response_mask=response_mask,
            method=phase_method,
            percentile=phase_percentile,
            min_phase_len=phase_min_len,
            max_phases=phase_max_K,
        )

        # Step 3: Extract texts, golden answers, reference solutions
        max_K = max(len(b) for b in boundaries_batch)
        questions = []
        phase_texts_batch = []
        golden_answers = []
        ref_solutions_batch = []
        full_responses = []
        data_sources = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0

            # Question
            question = ""
            if hasattr(data.batch, "prompts"):
                question = data.batch["prompts"][b]
            elif "prompts" in data.batch:
                question = data.batch["prompts"][b]
            questions.append(question)

            # Golden answer & data source
            gt = ""
            ds = "math"
            if hasattr(data.batch, "ground_truths"):
                gt = data.batch["ground_truths"][b]
            elif "ground_truths" in data.batch:
                gt = data.batch["ground_truths"][b]
            if hasattr(data.batch, "data_sources"):
                ds = data.batch["data_sources"][b]
            elif "data_sources" in data.batch:
                ds = data.batch["data_sources"][b]
            golden_answers.append(gt)
            data_sources.append(ds)

            # Reference solutions from SolutionBank
            refs = solution_bank.get_solutions(
                question, max_return=max_ref_solutions_in_prompt
            )
            ref_solutions_batch.append(refs)

            # Segment into phases
            phases = segment_response_into_phases(
                boundaries=boundaries_batch[b],
                response_length=resp_end,
                token_ids=input_ids[b],
                tokenizer=tokenizer,
            )
            phase_texts_batch.append([p.text for p in phases])

            # Full response text
            resp_start = active[0].item() if len(active) > 0 else 0
            full_text = tokenizer.decode(
                input_ids[b][resp_start:resp_end], skip_special_tokens=True
            )
            full_responses.append(full_text)

        # Step 4: Score phases with LLM judge
        logger.info(f"[ADPO] Scoring {sum(len(p) for p in phase_texts_batch)} phases with {judge.__class__.__name__}")
        phase_rewards_list = judge.score_phases(
            questions=questions,
            phase_texts=phase_texts_batch,
            golden_answers=golden_answers,
            reference_solutions=ref_solutions_batch,
            data_sources=data_sources,
        )
        logger.info("[ADPO] Judge scoring complete")

        # Step 5: Compute outcome rewards and feed correct solutions back
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

        added = solution_bank.add_correct_generations(
            questions=questions,
            responses=full_responses,
            rewards=outcome_rewards,
            ground_truths=golden_answers,
            reward_threshold=1.0,
        )
        if added > 0:
            logger.info(f"[ADPO] Added {added} correct solutions to bank")

        # Periodic save
        _step_count[0] += 1
        if _step_count[0] % solution_bank_save_freq == 0:
            solution_bank.save_to_jsonl(solution_bank_save_path)

        # Step 6: Build phase reward tensor
        device = response_mask.device
        phase_rewards = torch.zeros(batch_size, max_K, device=device)
        phase_mask_tensor = torch.zeros(batch_size, max_K, device=device)

        for b in range(batch_size):
            n_phases = len(phase_rewards_list[b])
            for k in range(n_phases):
                phase_rewards[b, k] = phase_rewards_list[b][k]
                phase_mask_tensor[b, k] = 1.0

        # Step 7: Phase advantages -> token advantages
        token_advantages = compute_adpo_phase_advantages(
            log_probs=log_probs,
            phase_rewards=phase_rewards,
            phase_mask=phase_mask_tensor,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            sigma=phase_sigma,
        )

        data.batch["advantages"] = token_advantages

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_reward = phase_rewards[phase_mask_tensor > 0].mean().item() if phase_mask_tensor.sum() > 0 else 0
            avg_outcome = np.mean(outcome_rewards)
            bank_stats = solution_bank.get_stats()
            logger.info(
                f"[ADPO] phases={avg_phases:.1f}, "
                f"phase_reward={avg_reward:.3f}, "
                f"outcome={avg_outcome:.3f}, "
                f"bank={bank_stats['n_solutions']} sols / {bank_stats['n_questions']} qs"
            )

        return data

    RayPPOTrainer.compute_advantage = adpo_compute_advantage
    logger.info(
        f"Patched RayPPOTrainer.compute_advantage with ADPO "
        f"(judge={judge_type}, endpoint={judge_endpoint!r}, bank={solution_bank})"
    )
