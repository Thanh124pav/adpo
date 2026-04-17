import torch
import time
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .base import BaseReward
from adpo.reward_functions import compute_score
from adpo.pure_entropy_algorithm import extract_attention_at_layer_hf

logger = logging.getLogger(__name__)

class AttentionReward(BaseReward):
    """Reward computer based on attention-flow between phases.

    Extensibility — three pluggable functions, each independently swappable:

        hidden_confluence_fn(hidden_states, boundaries, resp_end) → np.ndarray (m, m)
            FAST PATH: operates directly on hidden states (L, D) moved to CPU
            immediately after _partial_forward. Skips reconstruct_attention_at_layer
            entirely — GPU is freed before any heavy computation.
            Default: None (uses attention reconstruction path below).
            Set via: config.algorithm.attention_hidden_pca_components > 0
                   or: AttentionReward(config, hidden_confluence_fn=my_fn)

        confluence_fn(attn_weights, boundaries, resp_end) → np.ndarray (m, m)
            ATTENTION PATH (only used when hidden_confluence_fn is None):
            Converts raw (num_heads, L, L) attention to an inter-phase confluence
            matrix. Default: build_phase_attention_matrix (head-averaged mean).
            Set via: config.algorithm.attention_pca_components > 0
                   or: AttentionReward(config, confluence_fn=my_fn)

        influence_fn(phase_attn, norm_mode) → np.ndarray (m-1, m-1)
            Converts any (m, m) confluence matrix to the lower-triangular
            influence matrix A used to solve for phase rewards.
            Default: build_influence_matrix_A.

    Memory comparison (resp_len=2048, hidden_dim=4096, num_heads=32):
        Attention path: Q/K on GPU 2×16MB, attn on CPU 536MB (num_heads×L²×4B)
        Hidden PCA path: hs_cpu 32MB (L×D×4B), Gram 32MB (L×L×8B), then tiny

    All three can be swapped after construction (e.g., reward.confluence_fn = fn).
    """

    def __init__(self, config, confluence_fn=None, influence_fn=None,
                 hidden_confluence_fn=None):
        super().__init__()
        algo = config.algorithm
        self.attention_layer = getattr(algo, 'attention_layer', 21)
        self.attention_norm_mode = getattr(algo, 'attention_norm_mode', None)
        self.use_direct_attention = getattr(algo, 'attention_use_direct', False)

        self.correct_reward = getattr(algo, 'correct_reward', 1.0)
        self.incorrect_reward = getattr(algo, 'incorrect_reward', -1)
        self.partial_reward = getattr(algo, 'partial_reward', 0.1)

        # Solve mode: form b (default) vs form a (legacy)
        #   form b: r[0] = first_phase_reward (fixed), r_last as additive bias at last row
        #   form a: r[last] = r_last (fixed exactly), set attention_fixed_first_reward=null
        self.first_phase_reward = getattr(algo, 'attention_fixed_first_reward', 0.5)

        # --- hidden_confluence_fn (fast path, skips attention reconstruction) ---
        # Priority: constructor arg > config key > None (use attention path)
        if hidden_confluence_fn is not None:
            self.hidden_confluence_fn = hidden_confluence_fn
        else:
            n_hs_pca = getattr(algo, 'attention_hidden_pca_components', 0)
            if n_hs_pca > 0:
                self.hidden_confluence_fn = make_hidden_pca_confluence_fn(n_hs_pca)
                logger.info(
                    f"[AttentionReward] Fast path: hidden PCA, n_components={n_hs_pca}"
                )
            else:
                self.hidden_confluence_fn = None  # use attention path

        # --- confluence_fn (attention path, used only when hidden_confluence_fn=None) ---
        if confluence_fn is not None:
            self.confluence_fn = confluence_fn
        else:
            n_pca = getattr(algo, 'attention_pca_components', 0)
            if n_pca > 0:
                self.confluence_fn = make_pca_confluence_fn(n_pca)
                logger.info(
                    f"[AttentionReward] Attention PCA confluence, n_components={n_pca}"
                )
            else:
                self.confluence_fn = build_phase_attention_matrix

        if self.use_direct_attention:
            logger.info("[AttentionReward] Direct attention path enabled (requires eager attention)")

        self.influence_fn = influence_fn if influence_fn is not None \
            else build_influence_matrix_A

    def compute(self, boundaries_batch, response_mask, index, token_ids, hf_model, golden_answers, full_responses, data_sources,  **context):
        t_step5 = time.time()
        outcome_rewards = []
        batch_size, seq_len = response_mask.shape
        for b in range(batch_size):
            if golden_answers[b]:
                r = compute_score(
                    data_source=data_sources[b],
                    solution_str=full_responses[b],
                    ground_truth=golden_answers[b],
                )
            else:
                r = 0.0
            outcome_rewards.append(r) # tính outcom_rewards 

        # Map outcome to last-phase reward
        last_phase_rewards = []
        for r in outcome_rewards:
            if r >= 1.0:
                last_phase_rewards.append(self.correct_reward)
            elif r > 0.0:
                last_phase_rewards.append(self.partial_reward)
            else:
                last_phase_rewards.append(self.incorrect_reward)

        n_correct = sum(1 for r in outcome_rewards if r >= 1.0)
        print(
            f"[PureEntropy Outcomes] {n_correct}/{batch_size} correct, "
            f"last_phase_rewards[:8]={[f'{v:.2f}' for v in last_phase_rewards[:8]]}",
            flush=True,
        )

        phase_rewards_batch = []
        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0] 
            # active là toàn bộ câu (nonzero ý chỉ việc ko padding)
            if len(active) == 0:
                phase_rewards_batch.append(np.array([0.0]))
                continue
            resp_start = active[0].item()
            resp_end = active[-1].item() + 1
            resp_len = resp_end - resp_start 
            n_phases = len(boundaries_batch[b])
            if n_phases <= 1:
                phase_rewards_batch.append(np.array([last_phase_rewards[b]]))
                continue

            n_layers = hf_model.config.num_hidden_layers
            if self.attention_layer < 0: 
                self.attention_layer = int(n_layers * 3/4)
            hf_device = next(hf_model.parameters()).device # lấy device
            sub_ids = token_ids[b:b+1].to(hf_device) # [1, seq_len]
            pos_ids = torch.arange(seq_len, device=hf_device).unsqueeze(0) # [1, seq_len]

            # Shift boundaries to response-relative coords
            boundaries_relative = [bd - resp_start for bd in boundaries_batch[b]]
            resp_end_relative = resp_end - resp_start

            layer_L = self.attention_layer

            # ----------------------------------------------------------------
            # DIRECT ATTENTION PATH: use outputs.attentions from eager attention.
            # This skips hidden-state extraction and reconstruction entirely.
            # ----------------------------------------------------------------
            if self.use_direct_attention:
                with torch.no_grad():
                    try:
                        attn_full = extract_attention_at_layer_hf(
                            hf_model,
                            sub_ids,
                            layer_L,
                        )
                    except Exception as e:
                        logger.error(
                            f"[PureEntropy] Direct attention extraction failed for b={b}: {e}"
                        )
                        print(
                            f"[PureEntropy] ERROR: Direct attention extraction failed for b={b}: {e}",
                            flush=True,
                        )
                        phase_rewards_batch.append(np.full(n_phases, last_phase_rewards[b]))
                        del sub_ids
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                attn_weights = attn_full[0, :, resp_start:resp_end, resp_start:resp_end].contiguous()
                del attn_full, sub_ids

                phase_attn = self.confluence_fn(
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
                        f"direct_attention=True, resp_len={resp_len} (sliced from seq_len={seq_len}):\n"
                        f"{np.array2string(phase_attn, precision=6, suppress_small=True)}",
                        flush=True,
                    )

            # ----------------------------------------------------------------
            # FAST PATH: hidden_confluence_fn set → move hs to CPU immediately,
            # skip reconstruct_attention_at_layer entirely.
            # ----------------------------------------------------------------
            elif self.hidden_confluence_fn is not None:
                with torch.no_grad():
                    hs_full = _partial_forward(hf_model, sub_ids, pos_ids, layer_L)
                    hs_resp = hs_full[:, resp_start:resp_end, :]
                    del hs_full, sub_ids

                hs_cpu = hs_resp[0].float().cpu()  # (resp_len, hidden_dim)
                del hs_resp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    phase_attn = self.hidden_confluence_fn(
                        hs_cpu, boundaries_relative, resp_end_relative,
                    )
                except Exception as e:
                    logger.error(
                        f"[PureEntropy] hidden_confluence_fn failed for b={b}: {e}"
                    )
                    print(
                        f"[PureEntropy] ERROR: hidden_confluence_fn failed b={b}: {e}",
                        flush=True,
                    )
                    phase_rewards_batch.append(np.full(n_phases, last_phase_rewards[b]))
                    continue

                if b == 0:
                    print(
                        f"[PureEntropy PhaseAttn] resp=0: phase_attn ({phase_attn.shape}), "
                        f"hidden_confluence_fn={self.hidden_confluence_fn.__name__}, "
                        f"resp_len={resp_len}:\n"
                        f"{np.array2string(phase_attn, precision=6, suppress_small=True)}",
                        flush=True,
                    )

            # ----------------------------------------------------------------
            # ATTENTION PATH: reconstruct Q·K^T at chosen layer.
            # matmul_on_cpu=True keeps the (num_heads, L, L) tensor off GPU.
            # ----------------------------------------------------------------
            else:
                with torch.no_grad():
                    hs_full = _partial_forward(hf_model, sub_ids, pos_ids, layer_L)
                    # hs_full: (1, seq_len, hidden_dim)

                    # Slice to response tokens only
                    hs_resp = hs_full[:, resp_start:resp_end, :]
                    # (1, resp_len, hidden_dim) — still on GPU

                    del hs_full, sub_ids

                    pos_resp = torch.arange(
                        resp_start, resp_end, device=hf_device,
                    ).unsqueeze(0)

                    try:
                        attn_weights = reconstruct_attention_at_layer(
                            model=hf_model,
                            layer_idx=layer_L,
                            hidden_state=hs_resp,
                            position_ids=pos_resp,
                            output_dtype=torch.float32,
                            matmul_on_cpu=True,
                        )  # (num_heads, resp_len, resp_len) float32, CPU
                    except Exception as e:
                        logger.error(
                            f"[PureEntropy] Attention reconstruction failed b={b}: {e}"
                        )
                        print(
                            f"[PureEntropy] ERROR: Attention reconstruction failed b={b}: {e}",
                            flush=True,
                        )
                        phase_rewards_batch.append(
                            np.full(n_phases, last_phase_rewards[b])
                        )
                        del hs_resp, pos_resp
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    del hs_resp, pos_resp

                phase_attn = self.confluence_fn(
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
                        f"confluence_fn={self.confluence_fn.__name__}, "
                        f"resp_len={resp_len} (sliced from seq_len={seq_len}):\n"
                        f"{np.array2string(phase_attn, precision=6, suppress_small=True)}",
                        flush=True,
                    )

            # --- Influence matrix A: pluggable via self.influence_fn ---
            A = self.influence_fn(phase_attn, norm_mode=self.attention_norm_mode)
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
                fixed_first_reward=self.first_phase_reward,  # form b default
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
        max_K = max(len(r) for r in phase_rewards_batch)
        device = response_mask.device
        phase_rewards_tensor = torch.zeros(batch_size, max_K, device=device)
        phase_mask_tensor = torch.zeros(batch_size, max_K, device=device)
        for b, rewards in enumerate(phase_rewards_batch):
            k = len(rewards)
            phase_rewards_tensor[b, :k] = torch.tensor(rewards, dtype=torch.float32)
            phase_mask_tensor[b, :k] = 1.0
        return phase_rewards_tensor, phase_mask_tensor, {}

def _get_hf_model(_hf_model, model_path):
    if _hf_model[0] is None:
        _hf_model[0] = _load_hf_model(model_path)
    return _hf_model[0]

def _load_hf_model(model_path: str):
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
        attn_impls = ["flash_attention_2", "eager"]
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


def _partial_forward(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    target_layer: int,
) -> torch.Tensor:
    """Run embedding + layers 0..target_layer only (skip later layers + lm_head).

    This is much faster than a full forward with output_hidden_states=True
    because:
    1. We don't compute layers target_layer+1 .. n_layers-1
    2. We don't compute lm_head (vocab projection)
    3. We don't store hidden states for all layers

    Args:
        model: HuggingFace AutoModelForCausalLM.
        input_ids: (batch, seq_len) token IDs.
        position_ids: (batch, seq_len) position IDs.
        target_layer: Layer index (0-based). We need the INPUT to this layer,
                      which is the OUTPUT of layer target_layer-1.

    Returns:
        hidden_states: (batch, seq_len, hidden_dim) at the target layer.
    """
    # Use a forward hook to capture hidden states at the target layer,
    # then abort the forward pass early. This guarantees correctness
    # (causal mask, RoPE, etc. are all handled by the model itself)
    # while still saving compute on layers after target_layer.
    base = model.model if hasattr(model, "model") else model
    target_module = base.layers[target_layer]

    captured = {}

    def hook_fn(module, args, kwargs):
        # Pre-forward hook: capture the input to this layer, then abort
        # args[0] is hidden_states (the input to this layer)
        captured["hidden_states"] = args[0].detach()
        # Raise to abort forward pass early (skip remaining layers + lm_head)
        raise _EarlyStopForward()

    handle = target_module.register_forward_pre_hook(hook_fn, with_kwargs=True)
    try:
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=False,
        )
    except _EarlyStopForward:
        pass  # Expected: we aborted the forward early
    finally:
        handle.remove()

    return captured["hidden_states"]

class _EarlyStopForward(Exception):
    """Sentinel exception to abort forward pass after capturing hidden states."""
    pass

def extract_hidden_states_hf(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract hidden states at a specific layer using HuggingFace forward pass.

    Since vLLM rollouts don't provide hidden states, we run a forward pass
    through the HF model to obtain them.

    Args:
        model: HuggingFace AutoModelForCausalLM (already loaded).
        tokenizer: Tokenizer (for reference, not used directly here).
        input_ids: (batch, seq_len) token IDs.
        response_mask: (batch, seq_len) response token mask.
        layer_idx: Which layer's hidden states to extract (0-indexed).

    Returns:
        hidden_states: (batch, seq_len, hidden_dim) at specified layer.
        position_ids: (batch, seq_len) position IDs used.
    """
    device = next(model.parameters()).device
    batch_size, seq_len = input_ids.shape

    import time
    t0 = time.time()

    logger.info(
        f"[PureEntropy HiddenStates] Extracting hidden states at layer {layer_idx}, "
        f"batch_size={batch_size}, seq_len={seq_len}"
    )
    print(
        f"[PureEntropy HiddenStates] Extracting hidden states at layer {layer_idx}, "
        f"batch_size={batch_size}, seq_len={seq_len}",
        flush=True,
    )

    # Process in sub-batches to manage memory
    all_hidden_states = []
    sub_batch_size = max(1, min(4, batch_size))  # process 4 at a time

    for start_b in range(0, batch_size, sub_batch_size):
        end_b = min(start_b + sub_batch_size, batch_size)
        sub_ids = input_ids[start_b:end_b].to(device)

        # Build position_ids
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(end_b - start_b, -1)

        with torch.no_grad():
            # Partial forward: only run embedding + layers 0..layer_idx
            # instead of full model forward with output_hidden_states=True
            # (avoids computing layers layer_idx+1..n_layers and lm_head)
            hs = _partial_forward(model, sub_ids, pos_ids, layer_idx)
            # hs: (sub_batch, seq_len, hidden_dim)

        all_hidden_states.append(hs.cpu())

        # Free memory
        del hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[PureEntropy HiddenStates] Sub-batch {start_b}-{end_b}/{batch_size} done "
            f"({time.time() - t0:.1f}s elapsed)",
            flush=True,
        )

    hidden_states = torch.cat(all_hidden_states, dim=0)  # (batch, seq_len, hidden_dim)
    position_ids = torch.arange(seq_len, device='cpu').unsqueeze(0).expand(batch_size, -1)

    elapsed = time.time() - t0
    logger.info(
        f"[PureEntropy HiddenStates] Extracted: shape={list(hidden_states.shape)}, "
        f"dtype={hidden_states.dtype}, time={elapsed:.1f}s"
    )
    print(
        f"[PureEntropy HiddenStates] Extracted: shape={list(hidden_states.shape)}, "
        f"dtype={hidden_states.dtype}, time={elapsed:.1f}s",
        flush=True,
    )

    return hidden_states, position_ids

def reconstruct_attention_at_layer(
    model,
    layer_idx: int,
    hidden_state: torch.Tensor,
    position_ids: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    matmul_on_cpu: bool = True,
) -> torch.Tensor:
    """Reconstruct attention weights for one layer from its input hidden state.

    Delegates to reasoning_analysis/attention_analysis/reconstruct.py.

    Args:
        model: HuggingFace model.
        layer_idx: Layer index.
        hidden_state: (1, seq_len, hidden_dim).
        position_ids: (1, seq_len).
        output_dtype: dtype for returned attention matrix.
        matmul_on_cpu: If True (default), run the large Q·K^T matmul and
                       softmax on CPU.  Steps 1-5 (LayerNorm, projection,
                       QK-Norm, RoPE) still run on GPU.  This eliminates
                       the O(num_heads × seq²) GPU memory spike entirely.

    Returns:
        attn_weights: (num_heads, seq_len, seq_len).
                      On CPU when matmul_on_cpu=True.
    """
    from reasoning_analysis.attention_analysis.reconstruct import reconstruct_attention

    with torch.no_grad():
        attn = reconstruct_attention(
            model, layer_idx, hidden_state, position_ids,
            output_dtype=output_dtype,
            matmul_on_cpu=matmul_on_cpu,
        )

    return attn  # (num_heads, seq_len, seq_len)


# ---------------------------------------------------------------------------
# Hidden-State Confluence (fast path — no attention reconstruction)
# ---------------------------------------------------------------------------

def build_phase_confluence_from_hidden_pca(
    hidden_states: torch.Tensor,
    boundaries: List[int],
    resp_end: int,
    n_components: int = 32,
) -> np.ndarray:
    """Phase confluence from PCA-compressed hidden states.

    Completely bypasses attention reconstruction: operates on
    hidden_states (resp_len, D) already on CPU, never touches GPU again.

    Algorithm:
        1. Center token embeddings: hs_c = hs - mean(hs, axis=0)
        2. Gram matrix G = hs_c @ hs_c.T  (resp_len × resp_len)
           — cheaper than (D×D) covariance when resp_len < D
        3. Top-n_comp eigenvectors of G = left singular vectors of hs_c
        4. Mean-pool token scores per phase → phase embeddings (m, n_comp)
        5. Cosine similarity → (m, m) confluence matrix

    Memory (resp_len=2048, D=4096, n_comp=32):
        hs on CPU: 2048 × 4096 × 4B = 32 MB
        G on CPU:  2048 × 2048 × 8B = 32 MB   ← peak
        result:    m × m × 8B        ≈ tiny

    vs. attention path peak: num_heads × resp_len² × 4B = 536 MB (n_h=32)

    Args:
        hidden_states: (resp_len, hidden_dim) float tensor on CPU.
        boundaries:    Relative phase boundary positions.
        resp_end:      End of response (relative, exclusive).
        n_components:  Number of PCA components (default 32).

    Returns:
        phase_attn: (m, m) cosine-similarity confluence matrix, values in [0, 1].
    """
    hs = hidden_states.numpy() if isinstance(hidden_states, torch.Tensor) \
        else np.asarray(hidden_states, dtype=np.float64)
    hs = hs.astype(np.float64)
    resp_len, D = hs.shape
    n_comp = min(n_components, resp_len - 1, D)

    # Center
    hs_mean = hs.mean(axis=0, keepdims=True)   # (1, D)
    hs_c = hs - hs_mean                         # (resp_len, D)

    # Gram matrix in token-space (resp_len × resp_len)
    # When resp_len < D (common for sliced response windows), this is
    # cheaper than the feature-space covariance (D × D).
    G = hs_c @ hs_c.T                           # (resp_len, resp_len)
    G /= max(resp_len - 1, 1)

    # Eigen-decomposition (eigh: symmetric, ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    top_idx = np.argsort(eigenvalues)[::-1][:n_comp]
    # eigenvectors[:,i] are the left singular vectors (U) of hs_c
    token_scores = eigenvectors[:, top_idx]     # (resp_len, n_comp)

    # Log explained variance at DEBUG
    total_var = max(eigenvalues.sum(), 1e-12)
    expl_var = eigenvalues[top_idx].sum() / total_var
    logger.debug(
        f"[AttentionReward HiddenPCA] n_comp={n_comp}, "
        f"explained_var={expl_var:.3f}, resp_len={resp_len}, D={D}"
    )

    # Phase embeddings: mean-pool token scores per phase
    m = len(boundaries)
    phase_embs = np.zeros((m, n_comp), dtype=np.float64)
    for k in range(m):
        s = boundaries[k]
        e = boundaries[k + 1] if k + 1 < m else resp_end
        if e > s:
            phase_embs[k] = token_scores[s:e].mean(axis=0)

    # Cosine similarity matrix → shift to [0, 1]
    norms = np.linalg.norm(phase_embs, axis=1, keepdims=True).clip(min=1e-12)
    phase_embs_normed = phase_embs / norms
    sim = phase_embs_normed @ phase_embs_normed.T   # (m, m), range [-1, 1]
    return (sim + 1.0) / 2.0


def make_hidden_pca_confluence_fn(n_components: int = 32):
    """Factory: return a hidden_confluence_fn using PCA on hidden states.

    The returned function has signature:
        fn(hidden_states, boundaries, resp_end) → np.ndarray (m, m)
    and can be passed directly as hidden_confluence_fn.

    Args:
        n_components: Number of PCA components (default 32).

    Usage:
        # At construction (automatic if config key set)
        reward = AttentionReward(config,
                     hidden_confluence_fn=make_hidden_pca_confluence_fn(64))

        # Or swap after construction
        reward.hidden_confluence_fn = make_hidden_pca_confluence_fn(16)

        # Via config key (auto-selected in AttentionReward.__init__)
        # config.algorithm.attention_hidden_pca_components = 32
    """
    def _hidden_pca(hidden_states, boundaries, resp_end):
        return build_phase_confluence_from_hidden_pca(
            hidden_states, boundaries, resp_end, n_components=n_components,
        )
    _hidden_pca.__name__ = f"hidden_pca_confluence(n={n_components})"
    return _hidden_pca


# ---------------------------------------------------------------------------
# Phase-Level Attention Matrix Construction
# ---------------------------------------------------------------------------

def build_phase_attention_matrix(
    attn_weights: torch.Tensor,
    boundaries: List[int],
    resp_end: int,
) -> np.ndarray:
    """Build inter-phase attention matrix from token-level attention.

    Computes mean attention between tokens in each pair of phases,
    averaged across all attention heads.

    Args:
        attn_weights: (num_heads, seq_len, seq_len) attention matrix.
        boundaries: Phase boundary positions (absolute token indices).
        resp_end: End of response (exclusive).

    Returns:
        phase_attn: (m, m) matrix where m = len(boundaries).
            phase_attn[a][b] = mean attention from tokens in phase a
            to tokens in phase b (head-averaged).
    """
    m = len(boundaries)
    # Head-averaged attention.  mean() first to reduce num_heads dim,
    # then float() for numpy compatibility (bfloat16 has no numpy dtype).
    attn_avg = attn_weights.mean(dim=0).float().cpu().numpy()  # (seq_len, seq_len)

    # Compute phase spans
    phase_spans = []
    for k in range(m):
        s = boundaries[k]
        e = boundaries[k + 1] if k + 1 < m else resp_end
        phase_spans.append((s, e))

    # Build m x m phase attention matrix
    phase_attn = np.zeros((m, m), dtype=np.float64)
    for a in range(m):
        a_start, a_end = phase_spans[a]
        for b in range(m):
            b_start, b_end = phase_spans[b]
            # Mean attention from phase a tokens (query) to phase b tokens (key)
            if a_end > a_start and b_end > b_start:
                block = attn_avg[a_start:a_end, b_start:b_end]
                phase_attn[a][b] = block.mean()

    return phase_attn


def build_phase_attention_matrix_pca(
    attn_weights: torch.Tensor,
    boundaries: List[int],
    resp_end: int,
    n_components: int = 1,
) -> np.ndarray:
    """PCA-reduced inter-phase attention matrix.

    Instead of simple head-averaging, aggregates the num_heads dimension
    via PCA.  Each (query, key) token-pair is treated as a point in
    num_heads-dimensional space.  PCA finds the directions of maximum
    variance across all token-pairs; the first n_components principal
    components capture the dominant shared attention pattern.

    Algorithm (avoids large SVD by working in head-space):
        X  : (L², H)   — each token-pair's head-attention vector
        C  : (H, H)    — covariance matrix in head-space (H << L²)
        eig(C) → top-n_comp eigenvectors → project → reconstruct → head-mean

    Why PCA helps:
        • Removes noise heads that fire randomly (PC variance → low → dropped)
        • Preserves heads with consistent, structured attention patterns
        • Reconstruction smooths out high-variance outlier heads

    Args:
        attn_weights: (num_heads, resp_len, resp_len) on CPU, float32.
        boundaries:   Relative phase boundary positions.
        resp_end:     End of response (relative, exclusive).
        n_components: Number of PCA components to keep (default 1).
                      1 → first PC only (most variance); higher values
                      preserve more head diversity.

    Returns:
        phase_attn: (m, m) inter-phase attention matrix.
    """
    num_heads, L, _ = attn_weights.shape
    n_comp = min(n_components, num_heads)

    # (H, L, L) → (L², H): each token-pair is a point in head-space
    X = attn_weights.float().cpu().numpy().reshape(num_heads, L * L).T  # (L², H)

    # Center
    X_mean = X.mean(axis=0, keepdims=True)   # (1, H)
    X_c = X - X_mean                          # (L², H)

    # Covariance in head-space: (H, H) — cheap since H (≤32..64) << L²
    n_samples = max(X_c.shape[0] - 1, 1)
    C = (X_c.T @ X_c) / n_samples            # (H, H)

    # Eigen-decomposition (eigh = symmetric, returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    top_idx = np.argsort(eigenvalues)[::-1][:n_comp]
    components = eigenvectors[:, top_idx].T   # (n_comp, H)

    # Project → reconstruct denoised X
    scores = X_c @ components.T               # (L², n_comp)
    X_approx = scores @ components + X_mean   # (L², H) denoised

    # Head-average the denoised matrix and reshape
    attn_denoised = X_approx.mean(axis=1).reshape(L, L)   # (L, L)
    attn_denoised = np.clip(attn_denoised, 0.0, None)     # attention ≥ 0

    # Explained variance ratio — log directly here
    total_var = max(eigenvalues.sum(), 1e-12)
    expl_var = eigenvalues[top_idx].sum() / total_var
    logger.debug(
        f"[AttentionReward PCA] n_components={n_comp}/{num_heads}, "
        f"explained_variance={expl_var:.3f}, L={L}"
    )

    # Build inter-phase matrix (same logic as base function)
    m = len(boundaries)
    phase_spans = []
    for k in range(m):
        s = boundaries[k]
        e = boundaries[k + 1] if k + 1 < m else resp_end
        phase_spans.append((s, e))

    phase_attn = np.zeros((m, m), dtype=np.float64)
    for a in range(m):
        a_s, a_e = phase_spans[a]
        for b in range(m):
            b_s, b_e = phase_spans[b]
            if a_e > a_s and b_e > b_s:
                phase_attn[a][b] = attn_denoised[a_s:a_e, b_s:b_e].mean()

    return phase_attn


def make_pca_confluence_fn(n_components: int = 1):
    """Factory: return a confluence_fn that uses PCA head-aggregation.

    The returned function has the same signature as build_phase_attention_matrix
    and can be passed directly as confluence_fn.

    Args:
        n_components: Number of PCA components (default 1 = first PC only).

    Usage:
        # At construction
        reward = AttentionReward(config, confluence_fn=make_pca_confluence_fn(3))

        # Or swap after construction
        reward.confluence_fn = make_pca_confluence_fn(n_components=5)

        # Via config key (auto-selected in AttentionReward.__init__)
        # config.algorithm.attention_pca_components = 3
    """
    def _pca_confluence(attn_weights, boundaries, resp_end):
        return build_phase_attention_matrix_pca(
            attn_weights, boundaries, resp_end, n_components=n_components,
        )
    _pca_confluence.__name__ = f"pca_confluence(n={n_components})"
    return _pca_confluence


def build_influence_matrix_A(
    phase_attn: np.ndarray,
    norm_mode: str = "row",
) -> np.ndarray:
    """Build the influence matrix A from the full phase attention matrix.

    A is (m-1) x (m-1), lower triangular (including diagonal).
    A[i][j] = phase_attn[i+1][j] (attention from phase (i+1) to phase j),
    representing the influence of phase j on phase (i+1).

    Key insight on the diagonal:
        A[i][i] = phase_attn[i+1][i] = attention from phase (i+1) to phase i.
        This is CROSS-phase attention (not self-attention), so it is kept.
        Self-attention (phase_attn[k][k]) is naturally excluded because
        A only uses rows 1..m-1 paired with cols 0..m-2.

    The equation: A * r[0:m-2] = r[1:m-1]
    Row i: sum_{j<=i} A[i][j] * r_j = r_{i+1}

    Args:
        phase_attn: (m, m) full phase attention matrix.
        norm_mode: Normalization mode for A after construction.
            "none" - No normalization (raw attention values).
            "row"  - Normalize each row to sum to 1.
                     r_{i+1} = weighted average of r_0..r_i.
            "col"  - Normalize each column to sum to 1.
                     Each source phase's total influence sums to 1.
            "matrix" - Normalize entire matrix to sum to 1.
                       Global normalization across all entries.

    Returns:
        A: (m-1, m-1) lower triangular matrix (diagonal included).
    """
    m = phase_attn.shape[0]
    if m < 2:
        return np.zeros((0, 0))

    n = m - 1  # A is n x n
    A = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            A[i][j] = phase_attn[i + 1][j]

    # Enforce lower triangular only (causal: phase i+1 can only attend to phases 0..i)
    # Diagonal A[i][i] = phase_attn[i+1][i] is valid cross-phase attention, keep it.
    for i in range(n):
        for j in range(i + 1, n):
            A[i][j] = 0.0  # zero upper triangle

    # Normalize
    eps = 1e-12
    if norm_mode == "row":
        for i in range(n):
            row_sum = A[i].sum()
            if row_sum > eps:
                A[i] /= row_sum
    elif norm_mode == "col":
        for j in range(n):
            col_sum = A[:, j].sum()
            if col_sum > eps:
                A[:, j] /= col_sum
    elif norm_mode == "matrix":
        mat_sum = A.sum()
        if mat_sum > eps:
            A /= mat_sum
    elif norm_mode == "none":
        pass
    else:
        raise ValueError(
            f"Unknown norm_mode={norm_mode!r}. "
            f"Expected one of: 'none', 'row', 'col', 'matrix'"
        )

    return A


def solve_phase_rewards(
    A: np.ndarray,
    r_last: float,
    n_phases: int,
    fixed_first_reward: Optional[float] = None,
) -> np.ndarray:
    """Solve for phase rewards.

    Modes:
        1. Default: solve A * r[0:m-2] = r[1:m-1] with the last reward fixed
           to r_last.
        2. Variant: if fixed_first_reward is provided, solve the shifted system
              with r[0] fixed to that value and solve for the remaining rewards
              using an explicit bias vector b.

    Given:
        A: (m-1, m-1) lower triangular matrix (diagonal included).
        r_last: reward of the last phase (r_{m-1}), from exact matching.
        n_phases: m, total number of phases.

    The system (0-indexed, x = [r_0, ..., r_{m-2}]):
        Row i (i=0..m-3): sum_{j<=i} A[i][j]*r_j = r_{i+1}  →  A[i]*x - r_{i+1} = 0
        Row m-2:          sum_{j<=m-2} A[m-2][j]*r_j = r_{m-1} (known)

    Rearrange into B*x = c:
        B = A, but B[i][i+1] -= 1 for i = 0..m-3
        c = [0, ..., 0, r_{m-1}]

    Example (m=4 phases, A is 3x3):
        Row 0: A[0][0]*r_0 = r_1             →  A[0][0]*r_0 - r_1 = 0
        Row 1: A[1][0]*r_0 + A[1][1]*r_1 = r_2  →  ... - r_2 = 0
        Row 2: A[2][0]*r_0 + A[2][1]*r_1 + A[2][2]*r_2 = r_3 (known)

    Returns:
        rewards: (m,) array of phase rewards [r_0, r_1, ..., r_{m-1}].
    """
    m = n_phases
    if m <= 1:
        if fixed_first_reward is not None:
            return np.array([fixed_first_reward]), 0.0
        return np.array([r_last]), 0.0

    n = m - 1  # size of the system
    if A.shape[0] == 0:
        if fixed_first_reward is not None:
            return np.array([fixed_first_reward]), 0.0
        return np.array([r_last]), 0.0

    if fixed_first_reward is not None:
        r0 = fixed_first_reward

        # Form b: fix r[0]=r0, inject r_last as additive bias at the last row.
        # System M @ y = rhs, where y = [r_1, ..., r_{m-1}]:
        #   row 0:    r_1 = A[0,0]*r_0
        #   row i>0:  r_{i+1} - sum_{j=1..i} A[i,j]*r_j = A[i,0]*r_0
        #   last row: rhs[-1] += r_last  (outcome injected as additive bias)
        # → r[m-1] = A[m-2]*r[0:m-1] + r_last  (biased, not fixed)
        M = np.zeros((n, n))
        M[0, 0] = 1.0
        for i in range(1, n):
            M[i, :i] = -A[i, 1:i + 1]
            M[i, i] = 1.0

        b = A[:, 0] * r0   # contribution from fixed r[0]
        b[-1] += r_last    # inject outcome reward as bias at last phase

        try:
            det_B = np.linalg.det(M)

            if abs(det_B) < 1e-12:
                logger.warning(
                    f"[PureEntropy Solve] B near-singular (det={det_B:.6e}), using lstsq"
                )
                y, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
            else:
                y = np.linalg.solve(M, b)
        except np.linalg.LinAlgError as e:
            logger.error(f"[PureEntropy Solve] LinAlgError: {e}, falling back to uniform")
            y = np.full(n, r0)

        rewards = np.zeros(m)
        rewards[0] = r0
        rewards[1:] = y

        reward_abs_max = max(abs(r0) * 10, abs(r_last) * 10, 5.0)
        rewards_clamped = np.clip(rewards, -reward_abs_max, reward_abs_max)
        if not np.allclose(rewards, rewards_clamped):
            logger.warning(
                f"[PureEntropy Solve] Clamped: {rewards} -> {rewards_clamped}"
            )
            rewards = rewards_clamped

        residual = 0.0
        if n > 0:
            residual = np.abs(M @ rewards[1:] - b).max()

        return rewards, residual

    # Build B = A (copy), then subtract 1 from superdiagonal for rows 0..n-2
    B = A.copy()
    for i in range(n - 1):
        B[i][i + 1] -= 1.0

    # Right-hand side
    c = np.zeros(n)
    c[-1] = r_last

    # Solve the system
    try:
        det_B = np.linalg.det(B)

        if abs(det_B) < 1e-12:
            logger.warning(
                f"[PureEntropy Solve] B near-singular (det={det_B:.6e}), using lstsq"
            )
            x, residuals, rank, sv = np.linalg.lstsq(B, c, rcond=None)
        else:
            x = np.linalg.solve(B, c)
    except np.linalg.LinAlgError as e:
        logger.error(f"[PureEntropy Solve] LinAlgError: {e}, falling back to uniform")
        x = np.full(n, r_last)

    # x = [r_0, r_1, ..., r_{m-2}]
    rewards = np.zeros(m)
    rewards[:n] = x
    rewards[-1] = r_last

    # Clamp extreme values (numerical stability)
    reward_abs_max = max(abs(r_last) * 10, 5.0)
    rewards_clamped = np.clip(rewards, -reward_abs_max, reward_abs_max)
    if not np.allclose(rewards, rewards_clamped):
        logger.warning(
            f"[PureEntropy Solve] Clamped: {rewards} -> {rewards_clamped}"
        )
        rewards = rewards_clamped

    # Verify: A * x should ≈ [r_1, ..., r_{m-1}]
    residual = 0.0
    if n > 0:
        lhs = A @ rewards[:n]
        rhs = rewards[1:]
        residual = np.abs(lhs - rhs).max()

    return rewards, residual