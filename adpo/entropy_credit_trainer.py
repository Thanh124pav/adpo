"""
Entropy-Credit Trainer -- monkey-patches verl's compute_advantage with
entropy-driven phase decomposition and entropy-based credit assignment.

No LLM-as-Judge, no attention reconstruction, no extra HF model.
Rewards come from exact matching (last phase) + entropy-based credit
for earlier phases. Much faster than pure-entropy (no forward pass needed).

Training flow:
1. Rollout (vLLM) → token IDs + log probs
2. Compute per-token entropy from log probs
3. Detect phase boundaries via sliding-window entropy percentile
4. Compute cumulative entropy per phase (with decay psi)
5. Determine threshold from correct responses in group
6. Phase reward = linear function of distance from threshold
7. Phase advantages → token advantages (with optional decay gamma)
"""

import logging
import traceback
import numpy as np
import torch
from typing import Optional

import verl.trainer.ppo.ray_trainer as ray_trainer_module

from adpo.entropy_credit_algorithm import (
    compute_token_entropy,
    detect_phase_boundaries_entropy_credit,
    segment_response_into_phases,
    compute_phase_cumulative_entropy,
    compute_entropy_credit_rewards,
    compute_entropy_credit_advantages,
)
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


def patch_verl_grpo_with_entropy_credit(
    tokenizer=None,
    # Phase detection params
    entropy_window_size: int = 10,
    entropy_percentile: float = 75.0,
    phase_min_len: int = 10,
    phase_max_K: int = 10,
    # Entropy credit params
    psi: float = 0.95,
    default_threshold_percentile: float = 90.0,
    # Total reward budget (sum of all phase rewards per response)
    correct_total: float = 1.0,
    incorrect_total: float = -1.0,
    partial_total: float = 0.1,
    # Advantage params
    decay_gamma: float = 0.0,
):
    """Monkey-patch verl's compute_advantage with entropy-credit algorithm.

    Args:
        tokenizer: HuggingFace tokenizer.
        entropy_window_size: Sliding window size k for entropy boundary detection.
        entropy_percentile: Percentile threshold for window means.
        phase_min_len: Minimum tokens per phase.
        phase_max_K: Maximum phases per response.
        psi: Decay factor for cumulative entropy (default 0.95).
        default_threshold_percentile: Percentile for threshold when all responses
            are wrong in a group (default 90.0).
        correct_total: Total reward budget when correct (default 1.0).
        incorrect_total: Total reward budget when wrong (default -1.0).
        partial_total: Total reward budget when partial/no answer (default 0.1).
        decay_gamma: In-phase advantage decay (0 = hard, >0 = decreasing).
    """
    original_compute_advantage = ray_trainer_module.compute_advantage

    def entropy_credit_compute_advantage(
        data, adv_estimator=None, gamma=1.0, lam=1.0,
        num_repeat=1, norm_adv_by_std_in_grpo=True, config=None,
    ):
        """Entropy-credit phase-based advantage computation."""
        print("[EntropyCredit] entropy_credit_compute_advantage called", flush=True)

        if tokenizer is None:
            print("[EntropyCredit] WARNING: No tokenizer — falling back", flush=True)
            return original_compute_advantage(
                data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
                num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        try:
            return _entropy_credit_inner(
                data, adv_estimator, gamma, lam,
                num_repeat, norm_adv_by_std_in_grpo, config,
            )
        except Exception as e:
            print(f"[EntropyCredit] ERROR: {e}", flush=True)
            traceback.print_exc()
            print("[EntropyCredit] Falling back to original compute_advantage", flush=True)
            return original_compute_advantage(
                data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
                num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

    def _entropy_credit_inner(
        data, adv_estimator, gamma_val, lam,
        num_repeat, norm_adv_by_std_in_grpo, config,
    ):
        """Core logic for entropy-credit advantage computation."""
        import time
        t0 = time.time()

        response_mask = data.batch["response_mask"]
        batch_size, seq_len = response_mask.shape
        log_probs = data.batch["old_log_probs"]
        device = response_mask.device

        # --- UID / Index ---
        index_raw = data.non_tensor_batch.get("uid", data.batch.get("uid", None))
        if isinstance(index_raw, np.ndarray):
            if index_raw.dtype == object:
                _, inverse = np.unique(index_raw, return_inverse=True)
                index = torch.tensor(inverse, dtype=torch.long, device=device)
            else:
                index = torch.tensor(index_raw, device=device)
        elif isinstance(index_raw, torch.Tensor):
            index = index_raw.to(device)
        else:
            index = torch.arange(batch_size, device=device)

        # --- Token IDs ---
        response_ids = data.batch.get("responses", None)
        input_ids = data.batch.get("input_ids", None)
        prompt_ids = data.batch.get("prompts", None)

        if response_ids is not None and response_ids.shape[1] == seq_len:
            token_ids = response_ids
        elif input_ids is not None and input_ids.shape[1] == seq_len:
            token_ids = input_ids
        else:
            token_ids = None

        if token_ids is None:
            print("[EntropyCredit] WARNING: No matching token_ids — falling back", flush=True)
            return original_compute_advantage(
                data, adv_estimator=adv_estimator, gamma=gamma_val, lam=lam,
                num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        # =====================================================================
        # STEP 1: Compute entropy
        # =====================================================================
        logits = data.batch.get("logits", None)
        entropy = compute_token_entropy(
            log_probs=log_probs, logits=logits, response_mask=response_mask,
        )

        active_entropy = entropy[response_mask > 0]
        if active_entropy.numel() > 0:
            print(
                f"[EntropyCredit Entropy] mean={active_entropy.mean():.4f}, "
                f"std={active_entropy.std():.4f}, "
                f"min={active_entropy.min():.4f}, max={active_entropy.max():.4f}",
                flush=True,
            )

        # =====================================================================
        # STEP 2: Detect phase boundaries
        # =====================================================================
        boundaries_batch = detect_phase_boundaries_entropy_credit(
            entropy=entropy,
            response_mask=response_mask,
            window_size=entropy_window_size,
            percentile=entropy_percentile,
            min_phase_len=phase_min_len,
            max_phases=phase_max_K,
        )

        avg_phases = np.mean([len(b) for b in boundaries_batch])
        print(f"[EntropyCredit Phases] avg_phases={avg_phases:.1f}", flush=True)

        # =====================================================================
        # STEP 3: Extract questions, golden answers, full responses
        # =====================================================================
        questions = []
        golden_answers = []
        full_responses = []
        data_sources = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0
            resp_start = active[0].item() if len(active) > 0 else 0

            question = ""
            if hasattr(data, 'non_tensor_batch') and 'raw_prompt' in data.non_tensor_batch:
                raw_prompt = data.non_tensor_batch['raw_prompt'][b]
                if isinstance(raw_prompt, list):
                    for msg in raw_prompt:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            question = msg.get('content', '')
                            break
                elif isinstance(raw_prompt, str):
                    question = raw_prompt
            if not question.strip() and prompt_ids is not None:
                question = tokenizer.decode(prompt_ids[b].tolist(), skip_special_tokens=True)
            questions.append(question)

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

            full_text = tokenizer.decode(
                token_ids[b][resp_start:resp_end].tolist(), skip_special_tokens=True,
            )
            full_responses.append(full_text)

        # =====================================================================
        # STEP 4: Compute outcome rewards (exact match)
        # =====================================================================
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
        print(
            f"[EntropyCredit Outcomes] {n_correct}/{batch_size} correct",
            flush=True,
        )

        # =====================================================================
        # STEP 5: Compute cumulative entropy per phase
        # =====================================================================
        cum_entropy_batch = compute_phase_cumulative_entropy(
            entropy=entropy,
            response_mask=response_mask,
            boundaries_batch=boundaries_batch,
            psi=psi,
        )

        # =====================================================================
        # STEP 6: Compute entropy-based phase rewards
        # =====================================================================
        phase_rewards_batch = compute_entropy_credit_rewards(
            cum_entropy_batch=cum_entropy_batch,
            outcome_rewards=outcome_rewards,
            index=index,
            correct_total=correct_total,
            incorrect_total=incorrect_total,
            partial_total=partial_total,
            default_percentile=default_threshold_percentile,
        )

        # =====================================================================
        # STEP 7: Phase rewards -> phase advantages -> token advantages
        # =====================================================================
        token_advantages, phase_advantages, phase_rewards_t, phase_mask_t = \
            compute_entropy_credit_advantages(
                response_mask=response_mask,
                index=index,
                boundaries_batch=boundaries_batch,
                phase_rewards_batch=phase_rewards_batch,
                decay_gamma=decay_gamma,
            )

        data.batch["advantages"] = token_advantages
        if "returns" not in data.batch.keys():
            data.batch["returns"] = torch.zeros_like(token_advantages)

        # =====================================================================
        # DEMO: Response 0
        # =====================================================================
        if batch_size > 0:
            active0 = response_mask[0].nonzero(as_tuple=True)[0]
            resp_start0 = active0[0].item() if len(active0) > 0 else 0
            resp_end0 = active0[-1].item() + 1 if len(active0) > 0 else 0

            full_text_0 = tokenizer.decode(
                token_ids[0][resp_start0:resp_end0].tolist(), skip_special_tokens=True,
            )

            print(f"[EntropyCredit Demo] ===== Response 0 =====", flush=True)
            print(f"[EntropyCredit Demo] Question: \"{questions[0][:200]}\"", flush=True)
            print(f"[EntropyCredit Demo] Golden answer: \"{golden_answers[0]}\"", flush=True)
            R_total_0 = correct_total if outcome_rewards[0] >= 1.0 else (
                partial_total if outcome_rewards[0] > 0 else incorrect_total
            )
            print(f"[EntropyCredit Demo] Outcome: {outcome_rewards[0]:.2f}, R_total={R_total_0:.2f}", flush=True)
            print(f"[EntropyCredit Demo] Full response ({len(full_text_0)} chars):", flush=True)
            print(full_text_0[:500], flush=True)
            if len(full_text_0) > 500:
                print("...(truncated)", flush=True)

            # Phase details
            n_phases_0 = len(boundaries_batch[0])
            print(f"[EntropyCredit Demo] {n_phases_0} phases, boundaries={boundaries_batch[0]}", flush=True)

            phases_0 = segment_response_into_phases(
                boundaries=boundaries_batch[0],
                response_length=resp_end0,
                token_ids=token_ids[0],
                tokenizer=tokenizer,
            )
            for k, phase in enumerate(phases_0):
                preview = phase.text[:120].replace('\n', '\\n')
                print(f"  phase {k} (tok {phase.start_idx}-{phase.end_idx}): \"{preview}\"", flush=True)

            # Cumulative entropy, rewards, advantages
            cum_e_0 = cum_entropy_batch[0]
            r0 = phase_rewards_t[0][phase_mask_t[0] > 0]
            a0 = phase_advantages[0][phase_mask_t[0] > 0]

            print(f"[EntropyCredit Demo] Cum entropy (psi={psi}): {[f'{e:.2f}' for e in cum_e_0]}", flush=True)
            print(f"[EntropyCredit Demo] Phase rewards: {[f'{v:.4f}' for v in r0.tolist()]}", flush=True)
            print(f"[EntropyCredit Demo] Phase advantages: {[f'{v:.4f}' for v in a0.tolist()]}", flush=True)

            tok_adv_0 = token_advantages[0][response_mask[0] > 0]
            if tok_adv_0.numel() > 0:
                print(
                    f"[EntropyCredit Demo] Token adv: mean={tok_adv_0.mean():.4f}, "
                    f"std={tok_adv_0.std():.4f}, min={tok_adv_0.min():.4f}, max={tok_adv_0.max():.4f}",
                    flush=True,
                )

            # Group info
            idx0 = index[0].item()
            group0_mask = (index == idx0)
            group0_outcomes = [outcome_rewards[i] for i in range(batch_size) if group0_mask[i]]
            group0_correct = sum(1 for r in group0_outcomes if r >= 1.0)
            print(f"[EntropyCredit Demo] Group uid={idx0}: {group0_correct}/{len(group0_outcomes)} correct", flush=True)
            print(f"[EntropyCredit Demo] ===== End Response 0 =====", flush=True)

        # =====================================================================
        # Diagnostics (batch-level)
        # =====================================================================
        elapsed = time.time() - t0
        with torch.no_grad():
            valid_rewards = phase_rewards_t[phase_mask_t > 0]
            valid_adv = phase_advantages[phase_mask_t > 0]
            tok_adv_all = token_advantages[response_mask > 0]

            diag_msg = (
                f"[EntropyCredit Summary] phases={avg_phases:.1f}, "
                f"outcome={np.mean(outcome_rewards):.3f}, "
                f"phase_reward(mean={valid_rewards.mean():.3f}, std={valid_rewards.std():.3f}, "
                f"min={valid_rewards.min():.3f}, max={valid_rewards.max():.3f}), "
                f"phase_adv(mean={valid_adv.mean():.3f}, std={valid_adv.std():.3f}), "
                f"token_adv(mean={tok_adv_all.mean():.3f}, std={tok_adv_all.std():.3f}), "
                f"time={elapsed:.1f}s"
            )
            print(diag_msg, flush=True)
            logger.info(diag_msg)

        return data

    # Apply the patch
    ray_trainer_module.compute_advantage = entropy_credit_compute_advantage
    patch_msg = (
        f"[EntropyCredit] Patched verl compute_advantage "
        f"(window={entropy_window_size}, pct={entropy_percentile}, "
        f"psi={psi}, threshold_pct={default_threshold_percentile}, "
        f"decay_gamma={decay_gamma})"
    )
    print(patch_msg, flush=True)
    logger.info(patch_msg)


class EntropyCreditTaskRunner:
    """Ray-remote TaskRunner for entropy-credit algorithm."""

    def __init__(self):
        from verl.trainer.main_ppo import TaskRunner
        self._inner = TaskRunner()

    def run(self, config):
        from transformers import AutoTokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        algo = config.algorithm

        patch_verl_grpo_with_entropy_credit(
            tokenizer=tokenizer,
            entropy_window_size=algo.get("entropy_window_size", 10),
            entropy_percentile=algo.get("entropy_percentile", 75.0),
            phase_min_len=algo.get("phase_min_len", 10),
            phase_max_K=algo.get("phase_max_K", 10),
            psi=algo.get("psi", 0.95),
            default_threshold_percentile=algo.get("default_threshold_percentile", 90.0),
            correct_total=algo.get("correct_total", 1.0),
            incorrect_total=algo.get("incorrect_total", -1.0),
            partial_total=algo.get("partial_total", 0.1),
            decay_gamma=algo.get("decay_gamma", 0.0),
        )

        self._inner.run(config)
