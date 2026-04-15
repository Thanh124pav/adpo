import torch
import time
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .base import BaseReward
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)

class AttentionReward(BaseReward):
    def __init__(self, config):
        super().__init__()
        algo = config.algorithm
        self.attention_layer = getattr(algo, 'attention_layer', 21)
        self.attention_norm_mode = getattr(algo, 'attention_norm_mode', None)

        self.correct_reward = getattr(algo, 'correct_reward', 1.0)
        self.incorrect_reward = getattr(algo, 'incorrect_reward', -1)
        self.partial_reward = getattr(algo, 'partial_reward', 0.1)

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

            # --- Partial forward for this sample only ---
            n_layers = hf_model.config.num_hidden_layers
            if self.attention_layer < 0: 
                self.attention_layer = int(n_layers * 3/4)
            hf_device = next(hf_model.parameters()).device # lấy device
            sub_ids = token_ids[b:b+1].to(hf_device) # [1, seq_len]
            pos_ids = torch.arange(seq_len, device=hf_device).unsqueeze(0) # [1, seq_len]

            with torch.no_grad():
                layer_L = self.attention_layer
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
                    # Nếu ko tái tạo đc thì lấy điểm reward cuối làm điểm chung ??? 
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
            A = build_influence_matrix_A(phase_attn, norm_mode=self.attention_norm_mode)
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
    import sys
    import os
    # Add reasoning_analysis to path for importing reconstruct module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ra_path = os.path.join(project_root, "reasoning_analysis")
    if ra_path not in sys.path:
        sys.path.insert(0, ra_path)

    from attention_analysis.reconstruct import reconstruct_attention

    with torch.no_grad():
        attn = reconstruct_attention(
            model, layer_idx, hidden_state, position_ids,
            output_dtype=output_dtype,
            matmul_on_cpu=matmul_on_cpu,
        )

    return attn  # (num_heads, seq_len, seq_len)


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
) -> np.ndarray:
    """Solve for phase rewards using the system A * r[0:m-2] = r[1:m-1].

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
        return np.array([r_last]), 0.0

    n = m - 1  # size of the system
    if A.shape[0] == 0:
        return np.array([r_last]), 0.0

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