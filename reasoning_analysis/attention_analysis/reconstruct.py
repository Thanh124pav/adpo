"""
Reconstruct attention matrices from hidden states + model weights.

Supports: Qwen2.5, Qwen3, Llama3, DeepSeek-R1-Distill-Qwen.

The key insight: attention weights are computed from hidden states via
    h -> LayerNorm -> Q,K projection -> (optional QK-Norm) -> RoPE -> Attention

So given the hidden state INPUT to a layer + the layer's weights,
we can reconstruct the exact attention matrix without a full forward pass.

Architecture differences handled:
    - GQA (num_kv_heads < num_heads): Qwen2.5, Qwen3, Llama3
    - QK-Norm (q_norm, k_norm):       Qwen3 only
    - RoPE (rotary position embed):   All (standard rotate-half)
    - rotary_emb location:            self_attn or model level
    - scaling attribute:              Optional per-architecture
    - Sliding window attention:       Qwen2.5 (some variants)
    - rotary_emb API:                 Old (seq_len) and new (position_ids)
    - head_dim override:              DeepSeek / some Qwen configs
"""

import math
import inspect

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE helpers (standard implementation, same across Qwen/Llama/DeepSeek)
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotate the last dimension: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to Q and K.

    Handles varying cos/sin shapes from different transformers versions:
        - (batch, seq, head_dim)     → unsqueeze(1) for head dim
        - (1, 1, seq, head_dim)      → already broadcast-ready
        - (seq, head_dim)            → unsqueeze(0).unsqueeze(0)
    """
    # Ensure cos/sin have shape (batch, 1, seq, head_dim) for broadcasting
    # with Q/K shape (batch, num_heads, seq, head_dim)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0 if cos.dim() == 0 else 1)
        sin = sin.unsqueeze(0 if sin.dim() == 0 else 1)

    # If cos has shape (batch, seq, head_dim), insert head dim
    if cos.dim() == q.dim() - 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Architecture-aware helpers
# ---------------------------------------------------------------------------

def get_rotary_emb(model, attn_module):
    """Find rotary embedding module (may be on attention or model level)."""
    if hasattr(attn_module, "rotary_emb"):
        return attn_module.rotary_emb
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    raise AttributeError(
        "Cannot find rotary_emb on self_attn or model.model. "
        "This architecture may not be supported."
    )


def call_rotary_emb(rotary_emb_module, x, position_ids, seq_len):
    """Call rotary_emb with the correct API (old or new transformers).

    Old API (transformers < 4.38): rotary_emb(x, seq_len=seq_len)
    New API (transformers >= 4.38): rotary_emb(x, position_ids)
    """
    sig = inspect.signature(rotary_emb_module.forward)
    params = list(sig.parameters.keys())

    if "position_ids" in params:
        return rotary_emb_module(x, position_ids)
    elif "seq_len" in params:
        return rotary_emb_module(x, seq_len=seq_len)
    else:
        # Try new API first, fall back to old
        try:
            return rotary_emb_module(x, position_ids)
        except TypeError:
            return rotary_emb_module(x, seq_len=seq_len)


def get_head_dim(config):
    """Get head_dim, handling explicit head_dim configs (DeepSeek, some Qwen)."""
    if hasattr(config, "head_dim") and config.head_dim is not None:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


def get_sliding_window(config):
    """Get sliding window size if configured, else None."""
    sw = getattr(config, "sliding_window", None)
    if sw is not None and sw > 0:
        return sw
    # Some models store it differently
    sw = getattr(config, "max_window_layers", None)
    return None  # max_window_layers is layer count, not window size


# ---------------------------------------------------------------------------
# Core reconstruction
# ---------------------------------------------------------------------------

def reconstruct_attention(
    model,
    layer_idx: int,
    hidden_state: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct attention weights for one layer from its input hidden state.

    Args:
        model: HuggingFace AutoModelForCausalLM (eager attention).
        layer_idx: Layer index (0-based).
        hidden_state: (1, seq_len, hidden_dim) — input to this layer.
                      This is outputs.hidden_states[layer_idx] from a forward pass.
        position_ids: (1, seq_len) — position IDs for RoPE.
                      Defaults to [0, 1, ..., seq_len-1].

    Returns:
        attn_weights: (num_heads, seq_len, seq_len) float32 attention matrix.
    """
    config = model.config
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = get_head_dim(config)

    seq_len = hidden_state.shape[1]
    device = hidden_state.device
    dtype = hidden_state.dtype

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Step 1: Input LayerNorm (RMSNorm)
    h = layer.input_layernorm(hidden_state)

    # Step 2: Q, K linear projection
    # Works for both with/without bias (handled by nn.Linear)
    Q = attn.q_proj(h)  # (1, seq, num_heads * head_dim)
    K = attn.k_proj(h)  # (1, seq, num_kv_heads * head_dim)

    # Step 3: Reshape to (batch, num_heads, seq, head_dim)
    Q = Q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Step 4: QK-Norm (Qwen3 has this, others don't)
    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        Q = attn.q_norm(Q)
    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        K = attn.k_norm(K)

    # Step 5: RoPE — rotate Q and K based on position
    rotary_emb_module = get_rotary_emb(model, attn)
    cos, sin = call_rotary_emb(rotary_emb_module, K, position_ids, seq_len)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    # Step 6: GQA — repeat KV heads to match Q heads
    if num_kv_heads != num_heads:
        n_rep = num_heads // num_kv_heads
        K = K.repeat_interleave(n_rep, dim=1)

    # Step 7: Scaled dot-product attention scores
    if hasattr(attn, "scaling"):
        scale = attn.scaling
    else:
        scale = 1.0 / math.sqrt(head_dim)

    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Step 8: Causal mask + optional sliding window
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )

    # Sliding window: some Qwen2.5 variants use it for certain layers
    sliding_window = get_sliding_window(config)
    # max_window_layers: only layers below this index use sliding window (Qwen2.5)
    max_window_layers = getattr(config, "max_window_layers", None)
    use_sliding = (
        sliding_window is not None
        and max_window_layers is not None
        and layer_idx < max_window_layers
    )
    if use_sliding:
        # Mask out positions beyond sliding window
        window_mask = torch.tril(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=-(sliding_window + 1),
        )
        causal_mask = causal_mask + window_mask

    attn_weights = attn_weights + causal_mask

    # Step 9: Softmax (in float32 for numerical stability)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    return attn_weights[0]  # (num_heads, seq_len, seq_len)
