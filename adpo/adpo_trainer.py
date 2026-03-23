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

import verl.trainer.ppo.ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo import core_algos

from adpo.adpo_algorithm import (
    compute_neg_log_probs,
    compute_token_entropy,
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

        # Step 1b: Compute entropy if needed for boundary detection
        entropy = None
        if self.phase_method == "entropy":
            logits = data.batch.get("logits", None)
            entropy = compute_token_entropy(
                log_probs=log_probs, logits=logits, response_mask=response_mask,
            )

        # Step 2: Detect phase boundaries
        boundaries_batch = detect_phase_boundaries(
            neg_log_probs=neg_log_probs,
            response_mask=response_mask,
            method=self.phase_method,
            delta=self.phase_delta,
            percentile=self.phase_percentile,
            min_phase_len=self.phase_min_len,
            max_phases=self.phase_max_K,
            entropy=entropy,
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
        if "returns" not in data.batch.keys():
            data.batch["returns"] = torch.zeros_like(token_advantages)

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_outcome = np.mean(outcome_rewards)
            bank_stats = self.solution_bank.get_stats()

            # Phase-level reward statistics from LLM judge
            valid_rewards = phase_rewards[phase_mask_tensor > 0]
            if valid_rewards.numel() > 0:
                reward_mean = valid_rewards.mean().item()
                reward_std = valid_rewards.std().item() if valid_rewards.numel() > 1 else 0.0
                reward_min = valid_rewards.min().item()
                reward_max = valid_rewards.max().item()
            else:
                reward_mean = reward_std = reward_min = reward_max = 0.0

            # Per-response mean reward (mean of phase rewards within each response)
            phase_count = phase_mask_tensor.sum(dim=1).clamp(min=1)
            response_mean_rewards = (phase_rewards * phase_mask_tensor).sum(dim=1) / phase_count
            active_responses = (phase_mask_tensor.sum(dim=1) > 0)
            if active_responses.any():
                resp_reward_mean = response_mean_rewards[active_responses].mean().item()
                resp_reward_std = response_mean_rewards[active_responses].std().item() if active_responses.sum() > 1 else 0.0
            else:
                resp_reward_mean = resp_reward_std = 0.0

            logger.info(
                f"[ADPO] phases={avg_phases:.1f}, "
                f"outcome={avg_outcome:.3f}, "
                f"phase_reward(mean={reward_mean:.3f}, std={reward_std:.3f}, "
                f"min={reward_min:.3f}, max={reward_max:.3f}), "
                f"resp_reward(mean={resp_reward_mean:.3f}, std={resp_reward_std:.3f}), "
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
    judge_timeout: float = 120.0,
    judge_max_tokens: int = 256,
):
    """Monkey-patch verl's module-level compute_advantage function with ADPO phase
    decomposition + LLM-as-Judge + SolutionBank.

    This patches the compute_advantage function in verl.trainer.ppo.ray_trainer
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
    if judge_type in ("endpoint", "api"):
        judge_kwargs["timeout"] = judge_timeout
        judge_kwargs["max_tokens"] = judge_max_tokens
    if judge_type in ("endpoint", "api", "vllm"):
        judge_kwargs["max_ref_solutions_in_prompt"] = max_ref_solutions_in_prompt
    judge = create_judge(judge_type=judge_type, judge_model=judge_model, **judge_kwargs)
    solution_bank = SolutionBank(max_solutions_per_question=max_solutions_per_question)
    solution_bank.load_from_directory(solution_bank_dir)

    _step_count = [0]  # mutable counter for closure

    original_compute_advantage = ray_trainer_module.compute_advantage

    def adpo_compute_advantage(data, adv_estimator=None, gamma=1.0, lam=1.0,
                                num_repeat=1, norm_adv_by_std_in_grpo=True,
                                config=None):
        """ADPO phase-based advantage with LLM-as-Judge scoring."""
        nonlocal tokenizer

        print("[ADPO] adpo_compute_advantage called", flush=True)

        # Lazy-load tokenizer — not available via args, skip ADPO if missing
        if tokenizer is None:
            print("[ADPO] WARNING: No tokenizer available — falling back to original compute_advantage", flush=True)
            return original_compute_advantage(data, adv_estimator=adv_estimator,
                                               gamma=gamma, lam=lam,
                                               num_repeat=num_repeat,
                                               norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                               config=config)

        try:
            return _adpo_compute_advantage_inner(data, adv_estimator, gamma, lam,
                                                  num_repeat, norm_adv_by_std_in_grpo, config)
        except Exception as e:
            import traceback
            print(f"[ADPO] ERROR in adpo_compute_advantage: {e}", flush=True)
            traceback.print_exc()
            print("[ADPO] Falling back to original compute_advantage", flush=True)
            return original_compute_advantage(data, adv_estimator=adv_estimator,
                                               gamma=gamma, lam=lam,
                                               num_repeat=num_repeat,
                                               norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                               config=config)

    def _adpo_compute_advantage_inner(data, adv_estimator, gamma, lam,
                                       num_repeat, norm_adv_by_std_in_grpo, config):
        """Inner logic — separated so exceptions are caught and logged."""
        response_mask = data.batch["response_mask"]
        batch_size, seq_len = response_mask.shape
        log_probs = data.batch["old_log_probs"]

        index_raw = data.non_tensor_batch.get("uid", data.batch.get("uid", None))
        if isinstance(index_raw, np.ndarray):
            if index_raw.dtype == object:
                # String UIDs — map to integer group indices
                unique_vals, inverse = np.unique(index_raw, return_inverse=True)
                index = torch.tensor(inverse, dtype=torch.long, device=response_mask.device)
            else:
                index = torch.tensor(index_raw, device=response_mask.device)
        elif isinstance(index_raw, torch.Tensor):
            index = index_raw.to(response_mask.device)
        else:
            index = torch.arange(batch_size, device=response_mask.device)
        input_ids = data.batch.get("input_ids", None)

        if input_ids is None:
            print("[ADPO] WARNING: No input_ids in batch — falling back to original compute_advantage", flush=True)
            return original_compute_advantage(data, adv_estimator=adv_estimator,
                                               gamma=gamma, lam=lam,
                                               num_repeat=num_repeat,
                                               norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                               config=config)

        # Step 1: -log pi
        neg_log_probs = compute_neg_log_probs(log_probs, response_mask)

        # Step 1b: Compute entropy if needed for boundary detection
        entropy = None
        if phase_method == "entropy":
            logits = data.batch.get("logits", None)
            entropy = compute_token_entropy(
                log_probs=log_probs, logits=logits, response_mask=response_mask,
            )

        # Step 2: Detect phase boundaries
        boundaries_batch = detect_phase_boundaries(
            neg_log_probs=neg_log_probs,
            response_mask=response_mask,
            method=phase_method,
            percentile=phase_percentile,
            min_phase_len=phase_min_len,
            max_phases=phase_max_K,
            entropy=entropy,
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

            # Question — decode from prompt token IDs
            prompt_ids = data.batch.get("prompts", None)
            if prompt_ids is not None:
                question = tokenizer.decode(prompt_ids[b], skip_special_tokens=True)
            else:
                question = ""
            questions.append(question)

            # Golden answer & data source — stored in non_tensor_batch
            gt = ""
            ds = "math"
            if "reward_model" in data.non_tensor_batch:
                rm_info = data.non_tensor_batch["reward_model"]
                if isinstance(rm_info, (list, np.ndarray)):
                    gt = rm_info[b].get("ground_truth", "") if isinstance(rm_info[b], dict) else ""
                elif isinstance(rm_info, dict):
                    gt_list = rm_info.get("ground_truth", None)
                    if gt_list is not None:
                        gt = gt_list[b] if hasattr(gt_list, '__getitem__') else ""
            if "data_source" in data.non_tensor_batch:
                ds_arr = data.non_tensor_batch["data_source"]
                ds = ds_arr[b] if hasattr(ds_arr, '__getitem__') else str(ds_arr)
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
        if "returns" not in data.batch.keys():
            data.batch["returns"] = torch.zeros_like(token_advantages)

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_outcome = np.mean(outcome_rewards)
            bank_stats = solution_bank.get_stats()

            # Phase-level reward statistics from LLM judge
            valid_rewards = phase_rewards[phase_mask_tensor > 0]
            if valid_rewards.numel() > 0:
                reward_mean = valid_rewards.mean().item()
                reward_std = valid_rewards.std().item() if valid_rewards.numel() > 1 else 0.0
                reward_min = valid_rewards.min().item()
                reward_max = valid_rewards.max().item()
            else:
                reward_mean = reward_std = reward_min = reward_max = 0.0

            # Per-response mean reward (mean of phase rewards within each response)
            phase_count = phase_mask_tensor.sum(dim=1).clamp(min=1)
            response_mean_rewards = (phase_rewards * phase_mask_tensor).sum(dim=1) / phase_count
            active_responses = (phase_mask_tensor.sum(dim=1) > 0)
            if active_responses.any():
                resp_reward_mean = response_mean_rewards[active_responses].mean().item()
                resp_reward_std = response_mean_rewards[active_responses].std().item() if active_responses.sum() > 1 else 0.0
            else:
                resp_reward_mean = resp_reward_std = 0.0

            diag_msg = (
                f"[ADPO] phases={avg_phases:.1f}, "
                f"outcome={avg_outcome:.3f}, "
                f"phase_reward(mean={reward_mean:.3f}, std={reward_std:.3f}, "
                f"min={reward_min:.3f}, max={reward_max:.3f}), "
                f"resp_reward(mean={resp_reward_mean:.3f}, std={resp_reward_std:.3f}), "
                f"bank={bank_stats['n_solutions']} sols / {bank_stats['n_questions']} qs"
            )
            print(diag_msg, flush=True)
            logger.info(diag_msg)

        return data

    ray_trainer_module.compute_advantage = adpo_compute_advantage
    patch_msg = (
        f"[ADPO] Patched verl compute_advantage with ADPO "
        f"(judge={judge_type}, endpoint={judge_endpoint!r}, bank={solution_bank})"
    )
    print(patch_msg, flush=True)
    logger.info(patch_msg)


class ADPOTaskRunner:
    """Ray-remote TaskRunner that applies the ADPO monkey-patch inside the worker.

    verl's run_ppo spawns TaskRunner as a Ray remote actor (separate process).
    Monkey-patching in the driver has no effect — the patch must happen inside
    the worker process. This subclass does exactly that: in run(), it patches
    compute_advantage before delegating to the standard TaskRunner.run().
    """

    def __init__(self):
        from verl.trainer.main_ppo import TaskRunner
        self._inner = TaskRunner()

    def run(self, config):
        from transformers import AutoTokenizer
        from verl.utils.fs import copy_to_local

        # Download model and load tokenizer inside the worker
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        algo = config.algorithm
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
            judge_timeout=algo.get("judge_timeout", 120.0),
            judge_max_tokens=algo.get("judge_max_tokens", 256),
        )

        # Delegate to standard TaskRunner
        self._inner.run(config)
