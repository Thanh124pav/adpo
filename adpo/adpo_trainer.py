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
from adpo.reward_functions import compute_score
from adpo.golden_path import GoldenPathGenerator
from adpo.trajectory_stitching import TrajectoryStitcher, compute_stitched_advantages

logger = logging.getLogger(__name__)


class ADPOTrainer(RayPPOTrainer):
    """ADPO Trainer with phase-based advantage decomposition.

    Additional config keys (under algorithm):
        phase_method (str): "adaptive" or "threshold". Default "adaptive".
        phase_delta (float): Fixed threshold. Default 2.0.
        phase_percentile (float): Percentile threshold. Default 85.0.
        phase_min_len (int): Min tokens per phase. Default 10.
        phase_max_K (int): Max phases per response. Default 10.
        phase_decay_gamma (float): In-phase decay factor. Default 0.0 (off).
        incorrect_penalty (float): Score mapping scale for incorrect responses. Default 0.3.
        no_answer_correct_scale (float): Score scale for no-golden-answer good. Default 0.5.
        no_answer_incorrect_scale (float): Score scale for no-golden-answer bad. Default 0.1.
        judge_type (str): "vllm", "api", or "rule". Default "rule".
        judge_model (str): Judge model. Default "Qwen/Qwen2.5-7B-Instruct".
    """

    def __init__(self, config, tokenizer=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        algo = config.algorithm
        self.phase_method = getattr(algo, "phase_method", "adaptive")
        self.phase_delta = getattr(algo, "phase_delta", 2.0)
        self.phase_percentile = getattr(algo, "phase_percentile", 85.0)
        self.phase_min_len = getattr(algo, "phase_min_len", 10)
        self.phase_max_K = getattr(algo, "phase_max_K", 10)
        self.phase_decay_gamma = getattr(algo, "phase_decay_gamma", 0.0)
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

        self.tokenizer = tokenizer

        logger.info(
            f"ADPO Trainer: method={self.phase_method}, judge={judge_type}"
        )

    def compute_advantage(self, data):
        """Phase-based advantage with LLM-as-Judge scoring."""
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
            token_ids=input_ids,
            tokenizer=self.tokenizer,
        )

        # Step 3: Extract texts, golden answers
        max_K = max(len(b) for b in boundaries_batch)
        questions = []
        phase_texts_batch = []
        golden_answers = []
        full_responses = []
        data_sources = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0

            # Question — decode from prompt portion of input_ids
            question = ""
            if input_ids is not None and self.tokenizer is not None:
                resp_start = active[0].item() if len(active) > 0 else seq_len
                prompt_token_ids = input_ids[b][:resp_start]
                question = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
            questions.append(question)

            # Golden answer & data source
            gt = ""
            ds = "math"
            if hasattr(data, "non_tensor_batch") and "reward_model" in data.non_tensor_batch:
                rm_info = data.non_tensor_batch["reward_model"]
                if isinstance(rm_info, (list, np.ndarray)) and isinstance(rm_info[b], dict):
                    gt = rm_info[b].get("ground_truth", "")
                elif isinstance(rm_info, dict):
                    gt_list = rm_info.get("ground_truth", None)
                    if gt_list is not None and hasattr(gt_list, '__getitem__'):
                        gt = gt_list[b]
            elif hasattr(data.batch, "ground_truths"):
                gt = data.batch["ground_truths"][b]
            if hasattr(data, "non_tensor_batch") and "data_source" in data.non_tensor_batch:
                ds_arr = data.non_tensor_batch["data_source"]
                ds = ds_arr[b] if hasattr(ds_arr, '__getitem__') else str(ds_arr)
            elif hasattr(data.batch, "data_sources"):
                ds = data.batch["data_sources"][b]
            golden_answers.append(gt)
            data_sources.append(ds)

            # Segment into phases
            phases = segment_response_into_phases(
                boundaries=boundaries_batch[b],
                response_length=resp_end,
                token_ids=input_ids[b] if input_ids is not None else None,
                tokenizer=self.tokenizer,
            )
            phase_texts_batch.append([p.text for p in phases])

            # Full response text
            full_text = ""
            if input_ids is not None and self.tokenizer is not None:
                resp_start = active[0].item() if len(active) > 0 else 0
                full_text = self.tokenizer.decode(
                    input_ids[b][resp_start:resp_end], skip_special_tokens=True
                )
            full_responses.append(full_text)

        # Debug: verify question extraction
        n_empty_q = sum(1 for q in questions if not q.strip())
        n_empty_gt = sum(1 for g in golden_answers if not g.strip())
        if questions:
            preview = questions[0][:100].replace('\n', '\\n')
            print(f"[ADPO Questions] batch={batch_size}, empty_q={n_empty_q}, "
                  f"empty_gt={n_empty_gt}, q[0]=\"{preview}...\"", flush=True)

        # Demo: print full response 0 and phase boundaries
        if full_responses:
            print(f"[ADPO Full Response 0] ({len(full_responses[0])} chars):", flush=True)
            print(full_responses[0], flush=True)
            print(f"[ADPO Full Response 0 END]", flush=True)

        if phase_texts_batch and phase_texts_batch[0]:
            demo_bounds = boundaries_batch[0]
            n_total = len(phase_texts_batch[0])
            print(f"[ADPO Phases] response=0, {n_total} phases, "
                  f"boundaries={demo_bounds}", flush=True)
            for k, text in enumerate(phase_texts_batch[0]):
                b_start = demo_bounds[k] if k < len(demo_bounds) else "?"
                b_end = demo_bounds[k+1] if k+1 < len(demo_bounds) else "end"
                preview = text[:150].replace('\n', '\\n')
                print(f"  phase {k} (tok {b_start}-{b_end}): \"{preview}\"", flush=True)

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

        # Step 7: Phase advantages -> token advantages
        token_advantages = compute_adpo_phase_advantages(
            log_probs=log_probs,
            phase_rewards=phase_rewards,
            phase_mask=phase_mask_tensor,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            decay_gamma=self.phase_decay_gamma,
        )

        data.batch["advantages"] = token_advantages
        if "returns" not in data.batch.keys():
            data.batch["returns"] = torch.zeros_like(token_advantages)

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_outcome = np.mean(outcome_rewards)

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

            # Score mapping diagnostics per tier
            m_correct = has_golden & (outcome_tensor >= 1.0)
            m_incorrect = has_golden & (outcome_tensor < 1.0)
            m_no_good = no_golden & (response_judge_scores >= group_mean_scores)
            m_no_bad = no_golden & (response_judge_scores < group_mean_scores)

            def _avg(mask):
                return response_mean_rewards[mask].mean().item() if mask.any() else 0.0

            logger.info(
                f"[ADPO] phases={avg_phases:.1f}, "
                f"outcome={avg_outcome:.3f}, "
                f"phase_reward(mean={reward_mean:.3f}, std={reward_std:.3f}, "
                f"min={reward_min:.3f}, max={reward_max:.3f}), "
                f"resp_reward(mean={resp_reward_mean:.3f}, std={resp_reward_std:.3f}), "
                f"score_map("
                f"correct={_avg(m_correct):.3f}[n={m_correct.sum().item()}], "
                f"incorrect={_avg(m_incorrect):.3f}[n={m_incorrect.sum().item()}], "
                f"no_ans_good={_avg(m_no_good):.3f}[n={m_no_good.sum().item()}], "
                f"no_ans_bad={_avg(m_no_bad):.3f}[n={m_no_bad.sum().item()}])"
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
    phase_decay_gamma: float = 0.0,
    judge_timeout: float = 120.0,
    judge_max_tokens: int = 256,
    incorrect_penalty: float = 0.3,
    no_answer_correct_scale: float = 0.5,
    no_answer_incorrect_scale: float = 0.1,
    correct_reward_floor: float = 0.5,
    overlong_buffer_len: int = 0,
    overlong_penalty_factor: float = 1.0,
    output_correct_reward: float = 0.5,
    output_incorrect_reward: float = 0.0,
    golden_path_enabled: bool = False,
    golden_path_endpoint: str = "",
    golden_path_model: str = "",
    golden_path_mode: str = "endpoint",
    golden_path_cache_dir: str = "data/golden_paths",
    golden_path_max_attempts: int = 3,
    stitch_reward_decay: float = 0.1,
    stitch_max_extensions: int = 5,
    stitch_splice_boost: float = 2.0,
    stitch_post_splice_decay: float = 0.9,
):
    """Monkey-patch verl's module-level compute_advantage function with ADPO phase
    decomposition + LLM-as-Judge.

    This patches the compute_advantage function in verl.trainer.ppo.ray_trainer
    so we have access to the full data.batch, which is required for:
    - Decoding input_ids into phase texts for the LLM judge
    - Reading prompts, ground_truths, data_sources from the batch

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
    judge = create_judge(judge_type=judge_type, judge_model=judge_model, **judge_kwargs)

    # Golden path generator for answer-only datasets
    golden_gen = None
    if golden_path_enabled:
        gp_endpoint = golden_path_endpoint or judge_endpoint
        gp_model = golden_path_model or judge_model
        golden_gen = GoldenPathGenerator(
            endpoint=gp_endpoint,
            model=gp_model,
            mode=golden_path_mode,
            cache_dir=golden_path_cache_dir,
            max_attempts=golden_path_max_attempts,
        )
        logger.info(f"[ADPO] Golden path generator enabled: mode={golden_path_mode}, model={gp_model}")

    # Trajectory stitcher (only active when golden path is enabled)
    stitcher = None
    if golden_path_enabled:
        stitch_endpoint = golden_path_endpoint or judge_endpoint
        stitch_model = golden_path_model or judge_model
        stitcher = TrajectoryStitcher(
            endpoint=stitch_endpoint,
            model=stitch_model,
            tokenizer=tokenizer,
            max_response_length=2048,  # will be updated from data
            reward_decay=stitch_reward_decay,
            max_golden_extensions=stitch_max_extensions,
        )
        logger.info(f"[ADPO] Trajectory stitcher enabled: splice_boost={stitch_splice_boost}")

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
                unique_vals, inverse = np.unique(index_raw, return_inverse=True)
                index = torch.tensor(inverse, dtype=torch.long, device=response_mask.device)
            else:
                index = torch.tensor(index_raw, device=response_mask.device)
        elif isinstance(index_raw, torch.Tensor):
            index = index_raw.to(response_mask.device)
        else:
            index = torch.arange(batch_size, device=response_mask.device)

        # Determine which tensor to use for token decoding:
        # response_mask matches batch['responses'] shape, NOT input_ids
        response_ids = data.batch.get("responses", None)
        prompt_ids = data.batch.get("prompts", None)
        input_ids = data.batch.get("input_ids", None)

        # Pick the token_ids tensor that matches response_mask shape
        if response_ids is not None and response_ids.shape[1] == seq_len:
            token_ids = response_ids
        elif input_ids is not None and input_ids.shape[1] == seq_len:
            token_ids = input_ids
        else:
            token_ids = None

        if token_ids is None:
            print("[ADPO] WARNING: No matching token_ids for response_mask — "
                  f"response_mask.shape={response_mask.shape}, "
                  f"responses.shape={response_ids.shape if response_ids is not None else None}, "
                  f"input_ids.shape={input_ids.shape if input_ids is not None else None}",
                  flush=True)
            return original_compute_advantage(data, adv_estimator=adv_estimator,
                                               gamma=gamma, lam=lam,
                                               num_repeat=num_repeat,
                                               norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                               config=config)

        # Debug info
        active0 = response_mask[0].nonzero(as_tuple=True)[0]
        resp_start0 = active0[0].item() if len(active0) > 0 else -1
        resp_end0 = active0[-1].item() + 1 if len(active0) > 0 else -1
        n_active = int(response_mask[0].sum().item())
        print(f"[ADPO Debug] SHAPES: response_mask={list(response_mask.shape)}, "
              f"old_log_probs={list(log_probs.shape)}, "
              f"input_ids={list(input_ids.shape) if input_ids is not None else None}, "
              f"responses={list(response_ids.shape) if response_ids is not None else None}, "
              f"prompts={list(prompt_ids.shape) if prompt_ids is not None else None}, "
              f"token_ids(selected)={list(token_ids.shape)}",
              flush=True)
        print(f"[ADPO Debug] response_mask[0]: first=1@{resp_start0}, last=1@{resp_end0-1}, "
              f"n_active={n_active}, total={response_mask.shape[1]}",
              flush=True)
        if tokenizer is not None and resp_start0 >= 0:
            sample_text = tokenizer.decode(
                token_ids[0, resp_start0:min(resp_start0+10, resp_end0)].tolist(),
                skip_special_tokens=False,
            )
            print(f"[ADPO Debug] token_ids[0][{resp_start0}:{resp_start0+10}]: {sample_text!r}", flush=True)
        # Also show what's at start of each tensor
        if tokenizer is not None:
            if prompt_ids is not None:
                # Find first non-padding token in prompt
                p_decoded = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=True)
                print(f"[ADPO Debug] prompts[0] decoded (skip_special): \"{p_decoded[:150]}\"", flush=True)
            if response_ids is not None:
                r_decoded = tokenizer.decode(response_ids[0, :20].tolist(), skip_special_tokens=False)
                print(f"[ADPO Debug] responses[0][:20]: {r_decoded!r}", flush=True)

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
            token_ids=token_ids,
            tokenizer=tokenizer,
        )

        # Step 3: Extract texts, golden answers
        max_K = max(len(b) for b in boundaries_batch)
        questions = []
        phase_texts_batch = []
        golden_answers = []
        full_responses = []
        data_sources = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0

            # Question — try raw_prompt first, fallback to decoding batch['prompts']
            question = ""
            if hasattr(data, 'non_tensor_batch') and 'raw_prompt' in data.non_tensor_batch:
                raw_prompt = data.non_tensor_batch['raw_prompt'][b]
                if isinstance(raw_prompt, list):
                    # Chat format: [{"role": "user", "content": "..."}]
                    for msg in raw_prompt:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            question = msg.get('content', '')
                            break
                elif isinstance(raw_prompt, str):
                    question = raw_prompt
            if not question.strip() and prompt_ids is not None and tokenizer is not None:
                question = tokenizer.decode(prompt_ids[b].tolist(), skip_special_tokens=True)
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

            # Segment into phases (use token_ids = responses tensor)
            phases = segment_response_into_phases(
                boundaries=boundaries_batch[b],
                response_length=resp_end,
                token_ids=token_ids[b],
                tokenizer=tokenizer,
            )
            phase_texts_batch.append([p.text for p in phases])

            # Full response text
            resp_start = active[0].item() if len(active) > 0 else 0
            full_text = tokenizer.decode(
                token_ids[b][resp_start:resp_end].tolist(), skip_special_tokens=True
            )
            full_responses.append(full_text)

        # Debug: verify question extraction
        n_empty_q = sum(1 for q in questions if not q.strip())
        n_empty_gt = sum(1 for g in golden_answers if not g.strip())
        if questions:
            preview = questions[0][:100].replace('\n', '\\n')
            print(f"[ADPO Questions] batch={batch_size}, empty_q={n_empty_q}, "
                  f"empty_gt={n_empty_gt}, q[0]=\"{preview}...\"", flush=True)

        # Demo: print full response 0 and phase boundaries
        if full_responses:
            print(f"[ADPO Full Response 0] ({len(full_responses[0])} chars):", flush=True)
            print(full_responses[0], flush=True)
            print(f"[ADPO Full Response 0 END]", flush=True)

        if phase_texts_batch and phase_texts_batch[0]:
            demo_bounds = boundaries_batch[0]
            n_total = len(phase_texts_batch[0])
            print(f"[ADPO Phases] response=0, {n_total} phases, "
                  f"boundaries={demo_bounds}", flush=True)
            for k, text in enumerate(phase_texts_batch[0]):
                b_start = demo_bounds[k] if k < len(demo_bounds) else "?"
                b_end = demo_bounds[k+1] if k+1 < len(demo_bounds) else "end"
                preview = text[:150].replace('\n', '\\n')
                print(f"  phase {k} (tok {b_start}-{b_end}): \"{preview}\"", flush=True)

        # Step 4: Compute outcome rewards first (needed for output phase reward)
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

        n_correct = sum(1 for r in outcome_rewards if r >= 1.0)
        # Show group structure for response 0
        idx0 = index[0].item()
        group0_mask = (index == idx0)
        group0_outcomes = [outcome_rewards[i] for i in range(batch_size) if group0_mask[i]]
        group0_correct = sum(1 for r in group0_outcomes if r >= 1.0)
        print(f"[ADPO Outcomes] {n_correct}/{batch_size} correct, "
              f"first 8 responses: {['✓' if r >= 1.0 else f'{r:.2f}' for r in outcome_rewards[:8]]}", flush=True)
        print(f"[ADPO Group0] uid={idx0}, size={len(group0_outcomes)}, "
              f"correct={group0_correct}/{len(group0_outcomes)}, "
              f"outcomes={['✓' if r >= 1.0 else f'{r:.2f}' for r in group0_outcomes]}", flush=True)

        # Step 4a: Generate golden paths for all-wrong groups (answer-only datasets)
        # golden_paths[b] = text of golden path for response b, or None
        golden_paths = [None] * batch_size
        if golden_gen is not None:
            # Find groups where ALL responses are wrong
            all_wrong_groups = set()
            for uid in torch.unique(index):
                gmask = (index == uid)
                group_outcomes = [outcome_rewards[i] for i in range(batch_size) if gmask[i]]
                if all(r < 1.0 for r in group_outcomes):
                    all_wrong_groups.add(uid.item())

            if all_wrong_groups:
                # Collect unique questions from all-wrong groups (one per group)
                gp_questions, gp_answers, gp_sources, gp_group_ids = [], [], [], []
                seen_groups = set()
                for b in range(batch_size):
                    uid = index[b].item()
                    if uid in all_wrong_groups and uid not in seen_groups:
                        seen_groups.add(uid)
                        gp_questions.append(questions[b])
                        gp_answers.append(golden_answers[b])
                        gp_sources.append(data_sources[b])
                        gp_group_ids.append(uid)

                # Generate golden paths
                gp_results = golden_gen.generate_golden_paths(
                    gp_questions, gp_answers, gp_sources
                )

                # Map back: every response in an all-wrong group gets the golden path
                gp_by_group = {}
                for uid, gp_text in zip(gp_group_ids, gp_results):
                    if gp_text is not None:
                        gp_by_group[uid] = gp_text
                for b in range(batch_size):
                    uid = index[b].item()
                    if uid in gp_by_group:
                        golden_paths[b] = gp_by_group[uid]

                n_gp = sum(1 for gp in gp_by_group.values() if gp is not None)
                print(f"[ADPO GoldenPath] all_wrong_groups={len(all_wrong_groups)}, "
                      f"golden_paths_generated={n_gp}", flush=True)

        # Store golden_paths in data for trajectory stitching (next step)
        # Will be used by the stitching algorithm later
        data._golden_paths = golden_paths

        # Step 4b: Determine which phases are thinking vs output
        # Algorithm guarantees: if </think> found, it's a boundary → last phase = output
        # Just check if response has </think> token to know if last phase is output
        from adpo.adpo_algorithm import _find_think_boundary
        think_ends = _find_think_boundary(token_ids, response_mask, tokenizer)

        think_phase_texts_batch = []
        output_phase_idx = []
        for b in range(batch_size):
            n_phases = len(phase_texts_batch[b])
            has_output = (think_ends[b] is not None and n_phases >= 2)
            if has_output:
                think_phase_texts_batch.append(phase_texts_batch[b][:-1])
                output_phase_idx.append(n_phases - 1)
            else:
                think_phase_texts_batch.append(phase_texts_batch[b])
                output_phase_idx.append(-1)

        n_has_output = sum(1 for oi in output_phase_idx if oi >= 0)
        print(f"[ADPO Think/Output] has_output={n_has_output}/{batch_size}, "
              f"resp0: output_idx={output_phase_idx[0]}, "
              f"n_phases={len(phase_texts_batch[0])}", flush=True)

        # Step 4c: Score thinking phases with LLM judge
        total_think_phases = sum(len(p) for p in think_phase_texts_batch)
        logger.info(f"[ADPO] Scoring {total_think_phases} thinking phases with {judge.__class__.__name__}")
        think_rewards_list = judge.score_phases(
            questions=questions,
            phase_texts=think_phase_texts_batch,
            golden_answers=golden_answers,
            data_sources=data_sources,
        )
        logger.info("[ADPO] Judge scoring complete")

        # Step 4d: Build full phase_rewards_list: thinking rewards + output reward
        phase_rewards_list = []
        for b in range(batch_size):
            rewards = list(think_rewards_list[b])
            if output_phase_idx[b] >= 0:
                # Output phase reward: outcome-based
                if outcome_rewards[b] >= 1.0:
                    rewards.append(output_correct_reward)
                else:
                    rewards.append(output_incorrect_reward)
            phase_rewards_list.append(rewards)

        # Debug
        if phase_rewards_list:
            r0 = phase_rewards_list[0]
            oi = output_phase_idx[0]
            print(f"[ADPO Rewards] resp=0: {len(r0)} phases, output_idx={oi}, "
                  f"rewards={[f'{v:.3f}' for v in r0[:6]]}{'...' if len(r0)>6 else ''}", flush=True)

        # Step 5b: Overlong reward shaping (DAPO-style soft penalty)
        overlong_rewards = [0.0] * batch_size
        if overlong_buffer_len > 0:
            max_resp_len = response_mask.shape[1]
            expected_len = max_resp_len - overlong_buffer_len
            for b in range(batch_size):
                resp_len = int(response_mask[b].sum().item())
                exceed_len = resp_len - expected_len
                penalty = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0.0)
                overlong_rewards[b] = penalty
            n_penalized = sum(1 for r in overlong_rewards if r < 0)
            avg_penalty = np.mean([r for r in overlong_rewards if r < 0]) if n_penalized else 0.0
            print(f"[ADPO Overlong] expected_len={expected_len}, buffer={overlong_buffer_len}, "
                  f"penalized={n_penalized}/{batch_size}, avg_penalty={avg_penalty:.4f}", flush=True)

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
        phase_count_s = phase_mask_tensor.sum(dim=1).clamp(min=1)
        response_judge_scores = (phase_rewards * phase_mask_tensor).sum(dim=1) / phase_count_s

        # For no-golden-answer: classify as "good"/"bad" by comparing to group mean
        group_mean_scores = torch.zeros(batch_size, device=device)
        for uid in torch.unique(index):
            gmask = (index == uid)
            group_mean_scores[gmask] = response_judge_scores[gmask].mean()

        # Build score_scale per response
        score_scale = torch.ones(batch_size, device=device)
        score_scale[has_golden & (outcome_tensor < 1.0)] = incorrect_penalty
        score_scale[no_golden & (response_judge_scores >= group_mean_scores)] = no_answer_correct_scale
        score_scale[no_golden & (response_judge_scores < group_mean_scores)] = no_answer_incorrect_scale

        phase_rewards = phase_rewards * score_scale.unsqueeze(1)

        # Floor phase rewards for correct responses
        correct_mask = has_golden & (outcome_tensor >= 1.0)
        if correct_mask.any():
            floor_mask = correct_mask.unsqueeze(1) & (phase_mask_tensor > 0)
            phase_rewards = torch.where(
                floor_mask,
                torch.clamp(phase_rewards, min=correct_reward_floor),
                phase_rewards,
            )

        # Step 6d: Add overlong penalty to all phase rewards (broadcast per response)
        if overlong_buffer_len > 0:
            overlong_tensor = torch.tensor(overlong_rewards, device=device)
            # Distribute penalty evenly across all phases of each response
            phase_rewards = phase_rewards + (overlong_tensor.unsqueeze(1) * phase_mask_tensor)

        # Debug: show phase rewards for response 0
        pr0 = phase_rewards[0][phase_mask_tensor[0] > 0]
        print(f"[ADPO PhaseRewards] resp=0 outcome={outcome_rewards[0]:.2f} "
              f"rewards={[f'{v:.4f}' for v in pr0.tolist()[:5]]}... "
              f"mean={pr0.mean():.4f} std={pr0.std():.4f}", flush=True)

        # Step 7: Phase advantages -> token advantages
        token_advantages = compute_adpo_phase_advantages(
            log_probs=log_probs,
            phase_rewards=phase_rewards,
            phase_mask=phase_mask_tensor,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            decay_gamma=phase_decay_gamma,
        )

        # Step 8: Trajectory stitching for all-wrong groups
        if stitcher is not None and hasattr(data, '_golden_paths'):
            golden_paths_list = data._golden_paths
            # Find all-wrong groups that have golden paths
            all_wrong_groups = set()
            for uid in torch.unique(index):
                gmask = (index == uid)
                group_outcomes = [outcome_rewards[i] for i in range(batch_size) if gmask[i]]
                if all(r < 1.0 for r in group_outcomes):
                    all_wrong_groups.add(uid.item())

            stitch_results_map = {}  # batch_idx -> SpliceResult
            n_stitched = 0

            for uid_val in all_wrong_groups:
                gmask = (index == uid_val)
                group_indices = [i for i in range(batch_size) if gmask[i]]

                # Check if golden path available for this group
                gp_text = golden_paths_list[group_indices[0]]
                if gp_text is None:
                    continue

                # Gather group data
                g_questions = [questions[i] for i in group_indices]
                g_phase_texts = [phase_texts_batch[i] for i in group_indices]
                g_answers = [golden_answers[i] for i in group_indices]
                g_sources = [data_sources[i] for i in group_indices]
                g_boundaries = [boundaries_batch[i] for i in group_indices]

                try:
                    results = stitcher.stitch_group(
                        questions=g_questions,
                        response_phase_texts=g_phase_texts,
                        golden_path=gp_text,
                        golden_answers=g_answers,
                        data_sources=g_sources,
                        response_boundaries=g_boundaries,
                    )
                    for local_idx, result in enumerate(results):
                        batch_idx = group_indices[local_idx]
                        stitch_results_map[batch_idx] = result
                        if result.verified:
                            n_stitched += 1
                except Exception as e:
                    logger.warning(f"[Stitch] Failed for group {uid_val}: {e}")

            if stitch_results_map:
                token_advantages = compute_stitched_advantages(
                    token_advantages=token_advantages,
                    response_mask=response_mask,
                    stitch_results=stitch_results_map,
                    index=index,
                    splice_boost=stitch_splice_boost,
                    post_splice_decay=stitch_post_splice_decay,
                )
                print(f"[ADPO Stitch] {n_stitched}/{len(stitch_results_map)} "
                      f"responses stitched successfully in "
                      f"{len(all_wrong_groups)} all-wrong groups", flush=True)

        data.batch["advantages"] = token_advantages
        if "returns" not in data.batch.keys():
            data.batch["returns"] = torch.zeros_like(token_advantages)

        # Diagnostics
        with torch.no_grad():
            avg_phases = np.mean([len(b) for b in boundaries_batch])
            avg_outcome = np.mean(outcome_rewards)

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

            # Score mapping diagnostics per tier
            m_correct = has_golden & (outcome_tensor >= 1.0)
            m_incorrect = has_golden & (outcome_tensor < 1.0)
            m_no_good = no_golden & (response_judge_scores >= group_mean_scores)
            m_no_bad = no_golden & (response_judge_scores < group_mean_scores)

            def _avg_s(mask):
                return response_mean_rewards[mask].mean().item() if mask.any() else 0.0

            diag_msg = (
                f"[ADPO] phases={avg_phases:.1f}, "
                f"outcome={avg_outcome:.3f}, "
                f"phase_reward(mean={reward_mean:.3f}, std={reward_std:.3f}, "
                f"min={reward_min:.3f}, max={reward_max:.3f}), "
                f"resp_reward(mean={resp_reward_mean:.3f}, std={resp_reward_std:.3f}), "
                f"score_map("
                f"correct={_avg_s(m_correct):.3f}[n={m_correct.sum().item()}], "
                f"incorrect={_avg_s(m_incorrect):.3f}[n={m_incorrect.sum().item()}], "
                f"no_ans_good={_avg_s(m_no_good):.3f}[n={m_no_good.sum().item()}], "
                f"no_ans_bad={_avg_s(m_no_bad):.3f}[n={m_no_bad.sum().item()}])"
            )
            print(diag_msg, flush=True)
            logger.info(diag_msg)

        return data

    ray_trainer_module.compute_advantage = adpo_compute_advantage
    patch_msg = (
        f"[ADPO] Patched verl compute_advantage with ADPO "
        f"(judge={judge_type}, endpoint={judge_endpoint!r})"
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
            phase_decay_gamma=algo.get("phase_decay_gamma", 0.0),
            judge_timeout=algo.get("judge_timeout", 120.0),
            judge_max_tokens=algo.get("judge_max_tokens", 256),
            incorrect_penalty=algo.get("incorrect_penalty", 0.3),
            no_answer_correct_scale=algo.get("no_answer_correct_scale", 0.5),
            no_answer_incorrect_scale=algo.get("no_answer_incorrect_scale", 0.1),
            correct_reward_floor=algo.get("correct_reward_floor", 0.5),
            overlong_buffer_len=algo.get("overlong_buffer_len", 0),
            overlong_penalty_factor=algo.get("overlong_penalty_factor", 1.0),
            output_correct_reward=algo.get("output_correct_reward", 0.5),
            output_incorrect_reward=algo.get("output_incorrect_reward", 0.0),
            golden_path_enabled=algo.get("golden_path_enabled", False),
            golden_path_endpoint=algo.get("golden_path_endpoint", ""),
            golden_path_model=algo.get("golden_path_model", ""),
            golden_path_mode=algo.get("golden_path_mode", "endpoint"),
            golden_path_cache_dir=algo.get("golden_path_cache_dir", "data/golden_paths"),
            golden_path_max_attempts=algo.get("golden_path_max_attempts", 3),
            stitch_reward_decay=algo.get("stitch_reward_decay", 0.1),
            stitch_max_extensions=algo.get("stitch_max_extensions", 5),
            stitch_splice_boost=algo.get("stitch_splice_boost", 2.0),
            stitch_post_splice_decay=algo.get("stitch_post_splice_decay", 0.9),
        )

        # Delegate to standard TaskRunner
        self._inner.run(config)
