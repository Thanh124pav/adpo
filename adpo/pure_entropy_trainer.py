"""
Pure-Entropy Trainer -- monkey-patches verl's compute_advantage with
entropy-driven phase decomposition and attention-based reward propagation.

No LLM-as-Judge: rewards come purely from exact matching (last phase)
propagated backward through the inter-phase attention influence matrix.

Training flow:
1. Rollout (vLLM) → token IDs + log probs
2. Compute per-token entropy from log probs (or logits if available)
3. Detect phase boundaries via sliding-window entropy percentile
4. Load HF model, run forward pass to get hidden states at layer L
5. Reconstruct attention at layer L, build inter-phase attention matrix A
6. Exact-match score for last phase → solve A*r = propagated rewards
7. Phase advantages → token advantages → policy update
"""

import logging
import traceback
import numpy as np
import torch
from typing import Optional

import verl.trainer.ppo.ray_trainer as ray_trainer_module

from adpo.pure_entropy_algorithm import (
    compute_token_entropy,
    detect_phase_boundaries_pure_entropy,
    segment_response_into_phases,
    build_phase_mask,
    build_phase_attention_matrix,
    build_influence_matrix_A,
    solve_phase_rewards,
    extract_hidden_states_hf,
    reconstruct_attention_at_layer,
    compute_pure_entropy_advantages,
)
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


def _load_hf_model(model_path: str):
    """Load HuggingFace model for hidden state extraction.

    Loads with eager attention (required for reconstruction) and bfloat16.
    Cached after first load.
    """
    from transformers import AutoModelForCausalLM
    logger.info(f"[PureEntropy] Loading HF model from {model_path} for attention reconstruction...")
    print(f"[PureEntropy] Loading HF model from {model_path} for attention reconstruction...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Required for attention reconstruction
        device_map="auto",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    logger.info(
        f"[PureEntropy] HF model loaded: {model.config.architectures}, "
        f"n_layers={n_layers}, hidden_size={model.config.hidden_size}"
    )
    print(
        f"[PureEntropy] HF model loaded: {model.config.architectures}, "
        f"n_layers={n_layers}, hidden_size={model.config.hidden_size}",
        flush=True,
    )
    return model


def patch_verl_grpo_with_pure_entropy(
    tokenizer=None,
    model_path: str = "",
    # Phase detection params
    entropy_window_size: int = 10,
    entropy_percentile: float = 75.0,
    phase_min_len: int = 10,
    phase_max_K: int = 10,
    # Attention reconstruction params
    attention_layer: int = -1,  # Which layer L to reconstruct attention at (-1 = auto)
    attention_norm_mode: str = "row",  # Normalization for A: "none", "row", "col", "matrix"
    # Reward params
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
    partial_reward: float = 0.1,
):
    """Monkey-patch verl's compute_advantage with pure-entropy algorithm.

    Args:
        tokenizer: HuggingFace tokenizer.
        model_path: Path to model (for loading HF model for hidden states).
        entropy_window_size: Sliding window size k for entropy boundary detection.
        entropy_percentile: Percentile threshold for entropy windows.
        phase_min_len: Minimum tokens per phase.
        phase_max_K: Maximum phases per response.
        attention_layer: Layer index L for attention reconstruction.
            -1 = auto (3/4 of total layers).
        attention_norm_mode: Normalization for influence matrix A.
            "none" - raw attention values.
            "row"  - each row sums to 1 (r_{i+1} = weighted avg of earlier r).
            "col"  - each column sums to 1 (each source phase's total influence = 1).
            "matrix" - entire matrix sums to 1.
        correct_reward: Reward for last phase when answer is correct.
        incorrect_reward: Reward for last phase when answer is wrong.
        partial_reward: Reward for last phase on partial match.
    """
    original_compute_advantage = ray_trainer_module.compute_advantage

    # Lazy-loaded HF model
    _hf_model = [None]  # mutable container for closure

    def _get_hf_model():
        if _hf_model[0] is None:
            _hf_model[0] = _load_hf_model(model_path)
        return _hf_model[0]

    def pure_entropy_compute_advantage(
        data, adv_estimator=None, gamma=1.0, lam=1.0,
        num_repeat=1, norm_adv_by_std_in_grpo=True, config=None,
    ):
        """Pure-entropy phase-based advantage computation."""
        print("[PureEntropy] pure_entropy_compute_advantage called", flush=True)

        if tokenizer is None:
            print("[PureEntropy] WARNING: No tokenizer — falling back to original", flush=True)
            return original_compute_advantage(
                data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
                num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        try:
            return _pure_entropy_inner(
                data, adv_estimator, gamma, lam,
                num_repeat, norm_adv_by_std_in_grpo, config,
            )
        except Exception as e:
            print(f"[PureEntropy] ERROR: {e}", flush=True)
            traceback.print_exc()
            print("[PureEntropy] Falling back to original compute_advantage", flush=True)
            return original_compute_advantage(
                data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
                num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

    def _pure_entropy_inner(
        data, adv_estimator, gamma, lam,
        num_repeat, norm_adv_by_std_in_grpo, config,
    ):
        """Core logic for pure-entropy advantage computation."""
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
            print("[PureEntropy] WARNING: No matching token_ids — falling back", flush=True)
            return original_compute_advantage(
                data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
                num_repeat=num_repeat, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        # --- Debug info ---
        active0 = response_mask[0].nonzero(as_tuple=True)[0]
        resp_start0 = active0[0].item() if len(active0) > 0 else -1
        resp_end0 = active0[-1].item() + 1 if len(active0) > 0 else -1
        print(
            f"[PureEntropy Debug] batch_size={batch_size}, seq_len={seq_len}, "
            f"resp0: [{resp_start0}, {resp_end0}), "
            f"token_ids.shape={list(token_ids.shape)}",
            flush=True,
        )

        # =====================================================================
        # STEP 1: Compute entropy
        # =====================================================================
        logits = data.batch.get("logits", None)
        entropy = compute_token_entropy(
            log_probs=log_probs, logits=logits, response_mask=response_mask,
        )

        # Log entropy stats
        active_entropy = entropy[response_mask > 0]
        if active_entropy.numel() > 0:
            ent_mean = active_entropy.mean().item()
            ent_std = active_entropy.std().item()
            ent_min = active_entropy.min().item()
            ent_max = active_entropy.max().item()
            print(
                f"[PureEntropy Entropy] mean={ent_mean:.4f}, std={ent_std:.4f}, "
                f"min={ent_min:.4f}, max={ent_max:.4f}",
                flush=True,
            )

        # =====================================================================
        # STEP 2: Detect phase boundaries (pure entropy, sliding window)
        # =====================================================================
        boundaries_batch = detect_phase_boundaries_pure_entropy(
            entropy=entropy,
            response_mask=response_mask,
            window_size=entropy_window_size,
            percentile=entropy_percentile,
            min_phase_len=phase_min_len,
            max_phases=phase_max_K,
        )

        avg_phases = np.mean([len(b) for b in boundaries_batch])
        print(f"[PureEntropy Phases] avg_phases={avg_phases:.1f}", flush=True)

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

            # Question
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

            # Golden answer & data source
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

            # Full response text
            full_text = tokenizer.decode(
                token_ids[b][resp_start:resp_end].tolist(), skip_special_tokens=True,
            )
            full_responses.append(full_text)

        # =====================================================================
        # STEP 4: Compute exact-match outcome for last phase
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

        # Map outcome to last-phase reward
        last_phase_rewards = []
        for r in outcome_rewards:
            if r >= 1.0:
                last_phase_rewards.append(correct_reward)
            elif r > 0.0:
                last_phase_rewards.append(partial_reward)
            else:
                last_phase_rewards.append(incorrect_reward)

        n_correct = sum(1 for r in outcome_rewards if r >= 1.0)
        print(
            f"[PureEntropy Outcomes] {n_correct}/{batch_size} correct, "
            f"last_phase_rewards[:8]={[f'{v:.2f}' for v in last_phase_rewards[:8]]}",
            flush=True,
        )

        # =====================================================================
        # STEP 5: Load HF model and extract hidden states at layer L
        # =====================================================================
        hf_model = _get_hf_model()
        n_layers = hf_model.config.num_hidden_layers

        # Determine layer index
        layer_L = attention_layer
        if layer_L < 0:
            layer_L = int(n_layers * 3 / 4)
        layer_L = min(layer_L, n_layers - 1)
        layer_L = max(layer_L, 0)

        print(f"[PureEntropy] Using layer L={layer_L} (total layers={n_layers})", flush=True)

        # Extract hidden states
        hidden_states, position_ids_hs = extract_hidden_states_hf(
            model=hf_model,
            tokenizer=tokenizer,
            input_ids=token_ids,
            response_mask=response_mask,
            layer_idx=layer_L,
        )

        # =====================================================================
        # STEP 6: Build attention matrix and solve for phase rewards
        # =====================================================================
        phase_rewards_batch = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            if len(active) == 0:
                phase_rewards_batch.append(np.array([0.0]))
                continue

            resp_start = active[0].item()
            resp_end = active[-1].item() + 1
            n_phases = len(boundaries_batch[b])

            if n_phases <= 1:
                # Single phase: just use last-phase reward
                phase_rewards_batch.append(np.array([last_phase_rewards[b]]))
                if b == 0:
                    print(
                        f"[PureEntropy] resp=0: single phase, reward={last_phase_rewards[b]:.4f}",
                        flush=True,
                    )
                continue

            # Get hidden states for this response (slice to response portion)
            hs_b = hidden_states[b:b+1, :, :].to(
                dtype=torch.bfloat16,
                device=next(hf_model.parameters()).device,
            )
            pos_b = position_ids_hs[b:b+1].to(device=next(hf_model.parameters()).device)

            # Reconstruct attention at layer L
            try:
                attn_weights = reconstruct_attention_at_layer(
                    model=hf_model,
                    layer_idx=layer_L,
                    hidden_state=hs_b,
                    position_ids=pos_b,
                )  # (num_heads, seq_len, seq_len)
            except Exception as e:
                logger.error(f"[PureEntropy] Attention reconstruction failed for b={b}: {e}")
                print(f"[PureEntropy] ERROR: Attention reconstruction failed for b={b}: {e}", flush=True)
                phase_rewards_batch.append(np.full(n_phases, last_phase_rewards[b]))
                continue

            # Build phase attention matrix (m x m)
            phase_attn = build_phase_attention_matrix(
                attn_weights=attn_weights,
                boundaries=boundaries_batch[b],
                resp_end=resp_end,
            )

            if b == 0:
                print(
                    f"[PureEntropy PhaseAttn] resp=0: phase_attn ({phase_attn.shape}):\n"
                    f"{np.array2string(phase_attn, precision=6, suppress_small=True)}",
                    flush=True,
                )

            # Build influence matrix A (m-1 x m-1)
            A = build_influence_matrix_A(phase_attn, norm_mode=attention_norm_mode)

            if b == 0:
                print(
                    f"[PureEntropy InfluenceA] resp=0: A ({A.shape}):\n"
                    f"{np.array2string(A, precision=6, suppress_small=True)}",
                    flush=True,
                )

            # Solve for phase rewards
            rewards = solve_phase_rewards(
                A=A,
                r_last=last_phase_rewards[b],
                n_phases=n_phases,
            )
            phase_rewards_batch.append(rewards)

            if b == 0:
                print(
                    f"[PureEntropy Rewards] resp=0: {n_phases} phases, "
                    f"rewards={[f'{v:.4f}' for v in rewards]}",
                    flush=True,
                )

            # Free GPU memory per-sample
            del attn_weights, hs_b, pos_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =====================================================================
        # STEP 7: Phase rewards -> phase advantages -> token advantages
        # =====================================================================
        token_advantages = compute_pure_entropy_advantages(
            log_probs=log_probs,
            response_mask=response_mask,
            index=index,
            boundaries_batch=boundaries_batch,
            phase_rewards_batch=phase_rewards_batch,
        )

        data.batch["advantages"] = token_advantages
        if "returns" not in data.batch.keys():
            data.batch["returns"] = torch.zeros_like(token_advantages)

        # =====================================================================
        # Diagnostics
        # =====================================================================
        with torch.no_grad():
            avg_outcome = np.mean(outcome_rewards)

            all_rewards = np.concatenate(phase_rewards_batch)
            reward_mean = all_rewards.mean()
            reward_std = all_rewards.std() if len(all_rewards) > 1 else 0.0

            adv_active = token_advantages[response_mask > 0]
            adv_mean = adv_active.mean().item() if adv_active.numel() > 0 else 0.0
            adv_std = adv_active.std().item() if adv_active.numel() > 1 else 0.0

            diag_msg = (
                f"[PureEntropy] phases={avg_phases:.1f}, "
                f"outcome={avg_outcome:.3f}, "
                f"phase_reward(mean={reward_mean:.3f}, std={reward_std:.3f}), "
                f"token_adv(mean={adv_mean:.3f}, std={adv_std:.3f}), "
                f"layer_L={layer_L}"
            )
            print(diag_msg, flush=True)
            logger.info(diag_msg)

        return data

    # Apply the patch
    ray_trainer_module.compute_advantage = pure_entropy_compute_advantage
    patch_msg = (
        f"[PureEntropy] Patched verl compute_advantage with pure-entropy algorithm "
        f"(window={entropy_window_size}, pct={entropy_percentile}, layer={attention_layer})"
    )
    print(patch_msg, flush=True)
    logger.info(patch_msg)


class PureEntropyTaskRunner:
    """Ray-remote TaskRunner for pure-entropy algorithm.

    Applies the monkey-patch inside the worker process, then delegates
    to verl's standard TaskRunner.
    """

    def __init__(self):
        from verl.trainer.main_ppo import TaskRunner
        self._inner = TaskRunner()

    def run(self, config):
        from transformers import AutoTokenizer
        from verl.utils.fs import copy_to_local

        # Load tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        algo = config.algorithm

        patch_verl_grpo_with_pure_entropy(
            tokenizer=tokenizer,
            model_path=local_path,
            # Phase detection
            entropy_window_size=algo.get("entropy_window_size", 10),
            entropy_percentile=algo.get("entropy_percentile", 75.0),
            phase_min_len=algo.get("phase_min_len", 10),
            phase_max_K=algo.get("phase_max_K", 10),
            # Attention reconstruction
            attention_layer=algo.get("attention_layer", -1),
            attention_norm_mode=algo.get("attention_norm_mode", "row"),
            # Reward
            correct_reward=algo.get("correct_reward", 1.0),
            incorrect_reward=algo.get("incorrect_reward", 0.0),
            partial_reward=algo.get("partial_reward", 0.1),
        )

        # Delegate to standard TaskRunner
        self._inner.run(config)
