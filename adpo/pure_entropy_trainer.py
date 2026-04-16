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
    extract_attention_at_layer_hf,
    solve_phase_rewards,
    reconstruct_attention_at_layer,
    compute_pure_entropy_advantages,
)
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


def _pick_best_gpu() -> torch.device:
    """Pick the GPU with the most free memory.

    This avoids loading the HF model on the same GPU as actor/rollout
    (which typically uses most of GPU 0's memory).

    If only 1 GPU is available, uses it anyway (will share memory).
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return torch.device("cpu")

    best_gpu = 0
    best_free = 0

    for i in range(n_gpus):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        print(
            f"[PureEntropy] GPU {i}: {torch.cuda.get_device_name(i)}, "
            f"free={free / 1e9:.1f}GB, used={used / 1e9:.1f}GB, total={total / 1e9:.1f}GB",
            flush=True,
        )
        if free > best_free:
            best_free = free
            best_gpu = i

    device = torch.device(f"cuda:{best_gpu}")
    print(f"[PureEntropy] Selected GPU {best_gpu} (most free: {best_free / 1e9:.1f}GB)", flush=True)
    return device


def _load_hf_model(model_path: str, use_direct_attention: bool = False):
    """Load HuggingFace model for hidden state extraction + attention reconstruction.

    Automatically selects the GPU with the most free memory to avoid OOM
    when sharing with actor/rollout. Falls back to CPU if no GPU available.

    Cached after first load.
    """
    from transformers import AutoModelForCausalLM
    import os

    logger.info(f"[PureEntropy] Loading HF model from {model_path}...")
    print(f"[PureEntropy] Loading HF model from {model_path}...", flush=True)

    # Force CUDA visibility even if Ray didn't allocate a GPU to this worker.
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        print(f"[PureEntropy] CUDA not available (CUDA_VISIBLE_DEVICES={visible}), "
              "resetting to make all GPUs visible", flush=True)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        torch.cuda.init()

    if torch.cuda.is_available():
        # Pick the GPU with the most free memory (avoid the one used by actor/rollout)
        device = _pick_best_gpu()
        attn_impls = ["eager"] if use_direct_attention else ["flash_attention_2", "eager"]
    else:
        device = torch.device("cpu")
        attn_impls = ["eager"]
        print("[PureEntropy] WARNING: No CUDA available, using CPU", flush=True)

    # Try flash_attention_2 first (fastest), fall back to eager
    for attn_impl in attn_impls:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map={"": device},
            )
            print(f"[PureEntropy] Loaded with attn_implementation={attn_impl}, device={device}", flush=True)
            break
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[PureEntropy] flash_attention_2 unavailable ({e}), falling back to eager", flush=True)
            else:
                raise
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
    use_direct_attention: bool = False,
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
        use_direct_attention: If True, read attention directly from the
            model forward pass with output_attentions=True. Requires eager
            attention implementation.
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
            _hf_model[0] = _load_hf_model(model_path, use_direct_attention=use_direct_attention)
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
        # STEP 5+6 (fused): Partial forward → reconstruct attention →
        #   build phase matrix → solve rewards, one sample at a time.
        #
        # Key optimizations vs. naive approach:
        # 1. flash_attention_2 for the forward pass (2-4x faster)
        # 2. Partial forward: only layers 0..L-1 (skip later layers + lm_head)
        # 3. Slice to response tokens before attention reconstruction:
        #    O(resp_len²) instead of O(seq_len²)
        # 4. Never store full-batch hidden states — process & discard per sample
        # =====================================================================
        import time
        t_step5 = time.time()

        hf_model = _get_hf_model()
        n_layers = hf_model.config.num_hidden_layers
        hf_device = next(hf_model.parameters()).device

        # Determine layer index
        layer_L = attention_layer
        if layer_L < 0:
            layer_L = int(n_layers * 3 / 4)
        layer_L = min(layer_L, n_layers - 1)
        layer_L = max(layer_L, 0)

        print(f"[PureEntropy] Using layer L={layer_L} (total layers={n_layers})", flush=True)

        phase_rewards_batch = []

        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            if len(active) == 0:
                phase_rewards_batch.append(np.array([0.0]))
                continue

            resp_start = active[0].item()
            resp_end = active[-1].item() + 1
            resp_len = resp_end - resp_start
            n_phases = len(boundaries_batch[b])

            if n_phases <= 1:
                phase_rewards_batch.append(np.array([last_phase_rewards[b]]))
                if b == 0:
                    print(
                        f"[PureEntropy] resp=0: single phase, reward={last_phase_rewards[b]:.4f}",
                        flush=True,
                    )
                continue

            # --- Partial forward for this sample only ---
            sub_ids = token_ids[b:b+1].to(hf_device)
            pos_ids = torch.arange(seq_len, device=hf_device).unsqueeze(0)

            if use_direct_attention:
                with torch.no_grad():
                    try:
                        attn_full = extract_attention_at_layer_hf(
                            hf_model,
                            sub_ids,
                            layer_L,
                        )
                    except Exception as e:
                        logger.error(f"[PureEntropy] Direct attention extraction failed for b={b}: {e}")
                        print(f"[PureEntropy] ERROR: Direct attention extraction failed for b={b}: {e}", flush=True)
                        phase_rewards_batch.append(np.full(n_phases, last_phase_rewards[b]))
                        del sub_ids
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                attn_weights = attn_full[0, :, resp_start:resp_end, resp_start:resp_end].contiguous()
                del attn_full, sub_ids
            else:
                with torch.no_grad():
                    from adpo.pure_entropy_algorithm import _partial_forward
                    hs_full = _partial_forward(hf_model, sub_ids, pos_ids, layer_L)
                    # hs_full: (1, seq_len, hidden_dim)

                    # --- Slice to response tokens only ---
                    # This reduces attention reconstruction from O(seq_len²)
                    # to O(resp_len²). For seq_len=5120, resp_len~2048: ~6x speedup.
                    hs_resp = hs_full[:, resp_start:resp_end, :]
                    pos_resp = torch.arange(
                        resp_start, resp_end, device=hf_device,
                    ).unsqueeze(0)

                    del hs_full, sub_ids

                    # --- Reconstruct attention (response tokens only) ---
                    # matmul_on_cpu=True: Steps 1-5 (LayerNorm, Q/K proj, RoPE)
                    # run on GPU, then Q/K move to CPU for the large matmul.
                    # GPU never allocates the O(num_heads × resp_len²) tensor.
                    try:
                        attn_weights = reconstruct_attention_at_layer(
                            model=hf_model,
                            layer_idx=layer_L,
                            hidden_state=hs_resp,
                            position_ids=pos_resp,
                            output_dtype=torch.float32,
                            matmul_on_cpu=True,
                        )  # (num_heads, resp_len, resp_len) float32, on CPU
                    except Exception as e:
                        logger.error(f"[PureEntropy] Attention reconstruction failed for b={b}: {e}")
                        print(f"[PureEntropy] ERROR: Attention reconstruction failed for b={b}: {e}", flush=True)
                        phase_rewards_batch.append(np.full(n_phases, last_phase_rewards[b]))
                        del hs_resp, pos_resp
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    del hs_resp, pos_resp

            # --- Build phase attention matrix ---
            # Boundaries are in absolute coords; shift to response-relative
            # since attn_weights is now (num_heads, resp_len, resp_len)
            boundaries_relative = [bd - resp_start for bd in boundaries_batch[b]]
            resp_end_relative = resp_end - resp_start

            # attn_weights is already on CPU (matmul_on_cpu=True)
            phase_attn = build_phase_attention_matrix(
                attn_weights=attn_weights,
                boundaries=boundaries_relative,
                resp_end=resp_end_relative,
            )

            del attn_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if b == 0:
                print(
                    f"[PureEntropy PhaseAttn] resp=0: phase_attn ({phase_attn.shape}), "
                    f"resp_len={resp_len} (sliced from seq_len={seq_len}):\n"
                    f"{np.array2string(phase_attn, precision=6, suppress_small=True)}",
                    flush=True,
                )

            # --- Build influence matrix A ---
            A = build_influence_matrix_A(phase_attn, norm_mode=attention_norm_mode)

            if b == 0:
                print(
                    f"[PureEntropy InfluenceA] resp=0: A ({A.shape}):\n"
                    f"{np.array2string(A, precision=6, suppress_small=True)}",
                    flush=True,
                )

            # --- Solve for phase rewards ---
            rewards, residual = solve_phase_rewards(
                A=A,
                r_last=last_phase_rewards[b],
                n_phases=n_phases,
            )
            phase_rewards_batch.append(rewards)

            if b == 0:
                # Build B for logging
                n = n_phases - 1
                B_demo = A.copy()
                for ii in range(n - 1):
                    B_demo[ii][ii + 1] -= 1.0
                det_B = np.linalg.det(B_demo)

                print(
                    f"[PureEntropy Solve] resp=0: {n_phases} phases, "
                    f"det(B)={det_B:.6e}, residual={residual:.6e}",
                    flush=True,
                )
                print(
                    f"[PureEntropy Solve] B matrix:\n"
                    f"{np.array2string(B_demo, precision=6, suppress_small=True)}",
                    flush=True,
                )

            # Progress log every 16 samples
            if (b + 1) % 16 == 0 or b == batch_size - 1:
                elapsed = time.time() - t_step5
                rate = (b + 1) / elapsed
                eta = (batch_size - b - 1) / rate if rate > 0 else 0
                print(
                    f"[PureEntropy Progress] {b+1}/{batch_size} samples, "
                    f"{elapsed:.1f}s elapsed, {rate:.1f} samples/s, ETA {eta:.0f}s",
                    flush=True,
                )

        print(
            f"[PureEntropy] Step 5+6 done: {batch_size} samples in {time.time()-t_step5:.1f}s",
            flush=True,
        )

        # =====================================================================
        # STEP 7: Phase rewards -> phase advantages -> token advantages
        # =====================================================================
        token_advantages, phase_advantages, phase_rewards_t, phase_mask_t = \
            compute_pure_entropy_advantages(
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
        # DEMO: Response 0 — full text, phases, A, rewards, advantages
        # (tham khảo adpo_trainer.py)
        # =====================================================================
        if batch_size > 0:
            active0 = response_mask[0].nonzero(as_tuple=True)[0]
            resp_start0 = active0[0].item() if len(active0) > 0 else 0
            resp_end0 = active0[-1].item() + 1 if len(active0) > 0 else 0

            # -- Full response text --
            full_text_0 = tokenizer.decode(
                token_ids[0][resp_start0:resp_end0].tolist(),
                skip_special_tokens=True,
            )
            print(f"[PureEntropy Demo] ===== Response 0 =====", flush=True)
            print(f"[PureEntropy Demo] Question: \"{questions[0][:200]}\"", flush=True)
            print(f"[PureEntropy Demo] Golden answer: \"{golden_answers[0]}\"", flush=True)
            print(f"[PureEntropy Demo] Outcome: {outcome_rewards[0]:.2f} "
                  f"(last_phase_reward={last_phase_rewards[0]:.2f})", flush=True)
            print(f"[PureEntropy Demo] Full response ({len(full_text_0)} chars):", flush=True)
            print(full_text_0[:500], flush=True)
            if len(full_text_0) > 500:
                print("...(truncated)", flush=True)

            # -- Phase boundaries & texts --
            n_phases_0 = len(boundaries_batch[0])
            demo_bounds = boundaries_batch[0]
            print(f"[PureEntropy Demo] {n_phases_0} phases, boundaries={demo_bounds}", flush=True)

            phases_0 = segment_response_into_phases(
                boundaries=demo_bounds,
                response_length=resp_end0,
                token_ids=token_ids[0],
                tokenizer=tokenizer,
            )
            for k, phase in enumerate(phases_0):
                b_start = demo_bounds[k] if k < len(demo_bounds) else "?"
                b_end = demo_bounds[k + 1] if k + 1 < len(demo_bounds) else "end"
                preview = phase.text[:150].replace('\n', '\\n')
                print(f"  phase {k} (tok {b_start}-{b_end}): \"{preview}\"", flush=True)

            # -- Phase rewards (solved) --
            r0 = phase_rewards_t[0][phase_mask_t[0] > 0]
            print(
                f"[PureEntropy Demo] Phase rewards: "
                f"{[f'{v:.4f}' for v in r0.tolist()]}",
                flush=True,
            )

            # -- Phase advantages --
            a0 = phase_advantages[0][phase_mask_t[0] > 0]
            print(
                f"[PureEntropy Demo] Phase advantages: "
                f"{[f'{v:.4f}' for v in a0.tolist()]}",
                flush=True,
            )

            # -- Token advantage stats for resp 0 --
            tok_adv_0 = token_advantages[0][response_mask[0] > 0]
            if tok_adv_0.numel() > 0:
                print(
                    f"[PureEntropy Demo] Token adv resp=0: "
                    f"mean={tok_adv_0.mean():.4f}, std={tok_adv_0.std():.4f}, "
                    f"min={tok_adv_0.min():.4f}, max={tok_adv_0.max():.4f}",
                    flush=True,
                )

            # -- Group info --
            idx0 = index[0].item()
            group0_mask = (index == idx0)
            group0_outcomes = [outcome_rewards[i] for i in range(batch_size) if group0_mask[i]]
            group0_correct = sum(1 for r in group0_outcomes if r >= 1.0)
            print(
                f"[PureEntropy Demo] Group uid={idx0}: "
                f"{group0_correct}/{len(group0_outcomes)} correct",
                flush=True,
            )
            print(f"[PureEntropy Demo] ===== End Response 0 =====", flush=True)

        # =====================================================================
        # Diagnostics (batch-level summary)
        # =====================================================================
        with torch.no_grad():
            avg_outcome = np.mean(outcome_rewards)

            valid_rewards = phase_rewards_t[phase_mask_t > 0]
            if valid_rewards.numel() > 0:
                reward_mean = valid_rewards.mean().item()
                reward_std = valid_rewards.std().item() if valid_rewards.numel() > 1 else 0.0
                reward_min = valid_rewards.min().item()
                reward_max = valid_rewards.max().item()
            else:
                reward_mean = reward_std = reward_min = reward_max = 0.0

            valid_adv = phase_advantages[phase_mask_t > 0]
            if valid_adv.numel() > 0:
                adv_mean = valid_adv.mean().item()
                adv_std = valid_adv.std().item() if valid_adv.numel() > 1 else 0.0
            else:
                adv_mean = adv_std = 0.0

            token_adv_active = token_advantages[response_mask > 0]
            tok_adv_mean = token_adv_active.mean().item() if token_adv_active.numel() > 0 else 0.0
            tok_adv_std = token_adv_active.std().item() if token_adv_active.numel() > 1 else 0.0

            diag_msg = (
                f"[PureEntropy Summary] phases={avg_phases:.1f}, "
                f"outcome={avg_outcome:.3f}, "
                f"phase_reward(mean={reward_mean:.3f}, std={reward_std:.3f}, "
                f"min={reward_min:.3f}, max={reward_max:.3f}), "
                f"phase_adv(mean={adv_mean:.3f}, std={adv_std:.3f}), "
                f"token_adv(mean={tok_adv_mean:.3f}, std={tok_adv_std:.3f}), "
                f"layer_L={layer_L}"
            )
            print(diag_msg, flush=True)
            logger.info(diag_msg)

        return data

    # Apply the patch
    ray_trainer_module.compute_advantage = pure_entropy_compute_advantage
    patch_msg = (
        f"[PureEntropy] Patched verl compute_advantage with pure-entropy algorithm "
        f"(window={entropy_window_size}, pct={entropy_percentile}, layer={attention_layer}, "
        f"direct_attention={use_direct_attention})"
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
            use_direct_attention=algo.get("attention_use_direct", False),
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
