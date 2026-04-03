"""
Pure-Entropy Algorithm: Phase decomposition and reward assignment
purely driven by token-level entropy and attention-based influence.

Core idea:
1. Segment response into phases using a sliding-window entropy criterion:
   position t starts a new phase if mean(entropy[t:t+k]) > percentile_threshold.
2. Last phase reward = exact matching score.
3. Build inter-phase attention matrix A at hidden-state layer L.
4. Solve A * r[0:m-2] = r[1:m-1] to propagate rewards backward.

No LLM-as-Judge -- rewards are derived from the final outcome and
attention-based influence propagation.
"""

import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase Boundary Detection (pure entropy, sliding window)
# ---------------------------------------------------------------------------

def detect_phase_boundaries_pure_entropy(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    window_size: int = 10,
    percentile: float = 75.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Detect phase boundaries purely from entropy using a sliding window.

    A position t is a candidate phase start if the mean entropy over the
    window [t, t+window_size) exceeds the given percentile of all such
    window means across the response.

    Args:
        entropy: (batch, seq_len) per-token entropy values.
        response_mask: (batch, seq_len) binary mask for response tokens.
        window_size: Number of tokens in the sliding window (k).
        percentile: Percentile threshold for window means.
        min_phase_len: Minimum tokens between consecutive boundaries.
        max_phases: Maximum number of phases per response.

    Returns:
        List of boundary lists (one per batch element).
    """
    batch_size = entropy.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        ent = entropy[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1
        resp_len = end - start

        if resp_len <= window_size:
            all_boundaries.append([start])
            continue

        # Compute sliding-window means
        ent_vals = ent[start:end].float().cpu().numpy()
        n_windows = resp_len - window_size + 1
        window_means = np.array([
            ent_vals[i:i + window_size].mean()
            for i in range(n_windows)
        ])

        # Percentile threshold
        threshold = np.percentile(window_means, percentile)

        # Find candidate positions (in absolute token indices)
        candidates = []
        for i in range(n_windows):
            if window_means[i] > threshold:
                candidates.append((start + i, window_means[i]))

        # Greedy selection respecting min_phase_len
        boundaries = [start]
        candidates.sort(key=lambda x: -x[1])  # highest first
        for pos, score in candidates:
            if len(boundaries) >= max_phases:
                break
            if all(abs(pos - bd) >= min_phase_len for bd in boundaries):
                boundaries.append(pos)

        boundaries.sort()
        all_boundaries.append(boundaries)

        # Log first response
        if b == 0:
            logger.info(
                f"[PureEntropy Boundaries] resp=0: resp_len={resp_len}, "
                f"window_size={window_size}, threshold={threshold:.4f}, "
                f"n_candidates={len(candidates)}, "
                f"n_phases={len(boundaries)}, boundaries={boundaries}"
            )
            print(
                f"[PureEntropy Boundaries] resp=0: resp_len={resp_len}, "
                f"window_size={window_size}, threshold={threshold:.4f}, "
                f"n_candidates={len(candidates)}, "
                f"n_phases={len(boundaries)}, boundaries={boundaries}",
                flush=True,
            )

    return all_boundaries


# ---------------------------------------------------------------------------
# Entropy computation from logits or log_probs
# ---------------------------------------------------------------------------

def compute_token_entropy(
    log_probs: torch.Tensor = None,
    logits: torch.Tensor = None,
    response_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-token entropy from logits or approximate from log_probs.

    If logits are provided, computes exact entropy:
        H(t) = -sum_v p(v|context) * log p(v|context)
    If only log_probs, uses -log_prob as approximation.
    """
    if logits is not None:
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1)
    elif log_probs is not None:
        entropy = -log_probs
    else:
        raise ValueError("Either logits or log_probs must be provided")

    if response_mask is not None:
        entropy = entropy * response_mask

    return entropy


# ---------------------------------------------------------------------------
# Hidden States Extraction (HF forward pass)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Attention Reconstruction (from hidden states + model weights)
# ---------------------------------------------------------------------------

def reconstruct_attention_at_layer(
    model,
    layer_idx: int,
    hidden_state: torch.Tensor,
    position_ids: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
    matmul_on_cpu: bool = True,
) -> torch.Tensor:
    """Reconstruct attention weights for one layer from its input hidden state.

    Delegates to reasoning_analysis/attention_analysis/reconstruct.py.

    Args:
        model: HuggingFace model.
        layer_idx: Layer index.
        hidden_state: (1, seq_len, hidden_dim).
        position_ids: (1, seq_len).
        output_dtype: dtype for returned attention matrix.  Default bfloat16
                      to halve GPU memory vs float32.
        matmul_on_cpu: If True (default), run the large Q·K^T matmul and
                       softmax on CPU.  Steps 1-5 (LayerNorm, projection,
                       QK-Norm, RoPE) still run on GPU.  This eliminates
                       the O(num_heads × seq²) GPU memory spike entirely.

    Returns:
        attn_weights: (num_heads, seq_len, seq_len) in output_dtype.
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


# ---------------------------------------------------------------------------
# Reward Computation via Linear System
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase Segmentation (shared with adpo_algorithm)
# ---------------------------------------------------------------------------

@dataclass
class PhaseSegment:
    """A contiguous segment (phase) of a response."""
    phase_id: int
    start_idx: int
    end_idx: int
    text: str = ""
    reward: float = 0.0


def segment_response_into_phases(
    boundaries: List[int],
    response_length: int,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[PhaseSegment]:
    """Convert boundary indices into PhaseSegment objects."""
    phases = []
    for k in range(len(boundaries)):
        start = boundaries[k]
        end = boundaries[k + 1] if k + 1 < len(boundaries) else response_length
        text = ""
        if token_ids is not None and tokenizer is not None:
            text = tokenizer.decode(token_ids[start:end].tolist(), skip_special_tokens=True)
        phases.append(PhaseSegment(
            phase_id=k, start_idx=start, end_idx=end, text=text,
        ))
    return phases


# ---------------------------------------------------------------------------
# Phase Mask and Advantage Assignment
# ---------------------------------------------------------------------------

def build_phase_mask(
    boundaries_batch: List[List[int]],
    seq_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build tensor mapping each token to its phase index.

    Returns:
        phase_ids: (batch, seq_len) integer tensor, -1 for non-response.
    """
    batch_size = response_mask.shape[0]
    device = response_mask.device
    phase_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)

    for b in range(batch_size):
        boundaries = boundaries_batch[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            continue
        resp_end = active[-1].item() + 1

        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else resp_end
            phase_ids[b, start:end] = k

    return phase_ids


def compute_phase_advantages_pure_entropy(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute advantages: local (within response) + global (across group).

    Same structure as ADPO: additive combination of local and global advantages.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    # Local advantages
    phase_count = phase_mask.sum(dim=1).clamp(min=1)
    mean_per_response = (phase_rewards * phase_mask).sum(dim=1) / phase_count
    sq_diff = ((phase_rewards - mean_per_response.unsqueeze(1)) ** 2) * phase_mask
    variance = sq_diff.sum(dim=1) / phase_count.clamp(min=1)
    local_stds = torch.sqrt(variance + eps)
    local_adv = ((phase_rewards - mean_per_response.unsqueeze(1))
                 / (local_stds.unsqueeze(1) + eps)) * phase_mask

    # Global advantages (GRPO-style)
    response_scores = (phase_rewards * phase_mask).sum(dim=1) / phase_count
    global_adv = torch.zeros_like(phase_rewards)

    for uid in torch.unique(index):
        group_mask = (index == uid)
        group_scores = response_scores[group_mask]
        group_mean = group_scores.mean()
        group_std = group_scores.std() if group_scores.numel() > 1 else torch.tensor(0.0, device=device)
        adv = (group_scores - group_mean) / (group_std + eps)
        global_adv[group_mask] = adv.unsqueeze(1) * phase_mask[group_mask]

    return (local_adv + global_adv) * phase_mask


def assign_phase_advantages_to_tokens(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Map phase-level advantages back to token-level (hard assignment)."""
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device
    token_advantages = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        for t in range(seq_len):
            k = phase_ids[b, t].item()
            if k >= 0:
                token_advantages[b, t] = phase_advantages[b, k]

    return token_advantages * response_mask


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def compute_pure_entropy_advantages(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    boundaries_batch: List[List[int]],
    phase_rewards_batch: List[np.ndarray],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Full pipeline: phase rewards -> phase advantages -> token advantages.

    Args:
        log_probs: (batch, seq_len) token log probs.
        response_mask: (batch, seq_len) response mask.
        index: (batch,) group indices (uid).
        boundaries_batch: List of boundary lists per response.
        phase_rewards_batch: List of reward arrays per response.
        eps: Numerical stability constant.

    Returns:
        token_advantages: (batch, seq_len) per-token advantages.
        phase_advantages: (batch, max_K) per-phase advantages.
        phase_rewards_tensor: (batch, max_K) per-phase rewards.
        phase_mask_tensor: (batch, max_K) phase mask.
    """
    batch_size, seq_len = response_mask.shape
    device = response_mask.device

    max_K = max(len(b) for b in boundaries_batch)

    # Build phase reward tensor
    phase_rewards = torch.zeros(batch_size, max_K, device=device)
    phase_mask_tensor = torch.zeros(batch_size, max_K, device=device)

    for b in range(batch_size):
        n_phases = len(phase_rewards_batch[b])
        for k in range(n_phases):
            phase_rewards[b, k] = float(phase_rewards_batch[b][k])
            phase_mask_tensor[b, k] = 1.0

    # Phase advantages
    phase_advantages = compute_phase_advantages_pure_entropy(
        phase_rewards=phase_rewards,
        phase_mask=phase_mask_tensor,
        index=index,
        eps=eps,
    )

    # Phase IDs per token
    phase_ids = build_phase_mask(boundaries_batch, seq_len, response_mask)

    # Token advantages
    token_advantages = assign_phase_advantages_to_tokens(
        phase_advantages=phase_advantages,
        phase_ids=phase_ids,
        response_mask=response_mask,
    )

    return token_advantages, phase_advantages, phase_rewards, phase_mask_tensor
