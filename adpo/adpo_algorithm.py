"""
ADPO: Advantage Decomposition by Phase-Level Log Probability

Core idea:
1. Segment a generated response into K phases using log-probability
   boundaries (tokens where -log pi spikes = reasoning step transitions).
2. Score each phase independently with LLM-as-Judge.
3. Compute per-phase advantages (GRPO-style normalization within groups).
4. Assign each token the advantage of its parent phase.

This replaces GRPO's uniform response-level advantage with fine-grained
phase-level credit assignment.
"""

import torch
import re
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhaseSegment:
    """A contiguous segment (phase) of a response."""
    phase_id: int
    start_idx: int      # inclusive
    end_idx: int         # exclusive
    text: str = ""
    reward: float = 0.0
    advantage: float = 0.0


# ---------------------------------------------------------------------------
# Phase Boundary Detection
# ---------------------------------------------------------------------------

def compute_neg_log_probs(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute -log pi masked to response tokens."""
    return (-log_probs) * response_mask


def detect_phase_boundaries_threshold(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float = 2.0,
    min_phase_len: int = 10,
) -> List[List[int]]:
    """Detect phase boundaries using fixed threshold on -log pi.

    boundary(t) = 1[-log pi(a_t | s_t) > delta]
    with minimum spacing of min_phase_len tokens.
    """
    batch_size = neg_log_probs.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        nlp = neg_log_probs[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        boundaries = [start]
        last_boundary = start
        for t in range(start + min_phase_len, end):
            if nlp[t].item() > delta and (t - last_boundary) >= min_phase_len:
                boundaries.append(t)
                last_boundary = t

        all_boundaries.append(boundaries)

    return all_boundaries


def _get_delimiter_token_ids(tokenizer) -> set:
    """Build set of token IDs that represent sentence delimiters.

    Encodes common delimiters (. ? ! \\n etc.) and collects all token IDs
    that contain them. Cached per tokenizer instance.
    """
    if not hasattr(tokenizer, '_adpo_delim_ids'):
        delimiters = [".\n", ". ", "? ", "! ", "...", ".\n\n", "?\n", "!\n", "\n", "\n\n"]
        delim_ids = set()
        for d in delimiters:
            ids = tokenizer.encode(d, add_special_tokens=False)
            delim_ids.update(ids)
        # Also check single-char tokens
        for char in [".", "?", "!", "\n"]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            delim_ids.update(ids)
        tokenizer._adpo_delim_ids = delim_ids
    return tokenizer._adpo_delim_ids


def _find_sentence_boundaries(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    min_sentence_len: int = 5,
) -> List[List[Tuple[int, int]]]:
    """Split response tokens into sentence spans by finding delimiter tokens.

    Compares token IDs directly against pre-computed delimiter token IDs.
    No per-token decoding needed.

    Sentences shorter than min_sentence_len are merged into the next sentence.

    Returns:
        List (per batch) of list of (start_idx, end_idx) token spans.
    """
    delim_ids = _get_delimiter_token_ids(tokenizer)

    batch_size = response_mask.shape[0]
    all_sentences = []

    for b in range(batch_size):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_sentences.append([(0, 0)])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        # Find delimiter positions by comparing token IDs directly
        ids = token_ids[b, start:end].tolist()
        break_positions = [i for i, tid in enumerate(ids) if tid in delim_ids]

        # Convert break positions to sentence spans
        raw_sentences = []
        sent_start = 0  # relative to `start`
        for bp in break_positions:
            sent_end = bp + 1  # exclusive, include the delimiter token
            if sent_end > sent_start:
                raw_sentences.append((start + sent_start, start + sent_end))
            sent_start = sent_end

        # Remaining tokens form the last sentence
        if sent_start < (end - start):
            raw_sentences.append((start + sent_start, end))

        # Merge short sentences into the next one
        sentences = []
        for s_start, s_end in raw_sentences:
            if sentences and (s_start - sentences[-1][0]) < min_sentence_len:
                # Previous sentence too short — extend it
                sentences[-1] = (sentences[-1][0], s_end)
            else:
                sentences.append((s_start, s_end))

        # Fallback: if no sentences found, whole response = 1 sentence
        if not sentences:
            sentences = [(start, end)]

        all_sentences.append(sentences)

    return all_sentences


def _find_think_boundary(token_ids, response_mask, tokenizer):
    """Find token index where </think> ends for each response in batch.

    Tries multiple strategies:
    1. Match token IDs from tokenizer.encode("</think>")
    2. Match single special token ID from vocab
    3. Decode and search string (fallback)

    Returns list of int or None (one per batch element).
    """
    batch_size = response_mask.shape[0]
    results = []

    # Strategy 1: encode the marker
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)

    # Strategy 2: check if </think> is a single token in vocab
    think_end_single = None
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        tid = tokenizer.convert_tokens_to_ids("</think>")
        if tid != tokenizer.unk_token_id:
            think_end_single = tid

    marker_len = len(think_end_ids)

    for b in range(batch_size):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            results.append(None)
            continue
        start = active[0].item()
        end = active[-1].item() + 1
        ids = token_ids[b, start:end].tolist()

        found = None

        # Try single special token first (most likely for Qwen3)
        if think_end_single is not None:
            for i, tid in enumerate(ids):
                if tid == think_end_single:
                    found = start + i + 1
                    break

        # Try multi-token match
        if found is None and marker_len > 0:
            for i in range(len(ids) - marker_len + 1):
                if ids[i:i + marker_len] == think_end_ids:
                    found = start + i + marker_len
                    break

        # Debug: log for first response
        if b == 0 and found is None:
            print(f"[ADPO Debug] </think> NOT FOUND in response 0. "
                  f"encode('</think>')={think_end_ids}, "
                  f"single_token_id={think_end_single}, "
                  f"response_len={end-start}", flush=True)
        elif b == 0:
            print(f"[ADPO Debug] </think> found at tok={found} "
                  f"(single_id={think_end_single})", flush=True)

        results.append(found)
    return results


def detect_phase_boundaries_adaptive(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 85.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[List[int]]:
    """Detect phase boundaries using sentence structure + mean -log pi.

    1. Split response into sentences by punctuation delimiters (. ? ! \\n ...).
    2. If <think>...</think> present, always create boundary at </think>.
       Percentile thresholds are computed separately for thinking vs answer.
    3. For each sentence, compute mean -log pi of the first 1/3 tokens.
    4. If that mean > percentile threshold → sentence starts a new phase.
    5. Otherwise → sentence merges into the previous phase.

    Falls back to peak-based detection if token_ids/tokenizer not available.
    """
    if token_ids is None or tokenizer is None:
        return _detect_phase_boundaries_peak(
            neg_log_probs, response_mask, percentile, min_phase_len, max_phases,
        )

    batch_size = neg_log_probs.shape[0]
    sentences_batch = _find_sentence_boundaries(token_ids, response_mask, tokenizer)
    think_boundaries = _find_think_boundary(token_ids, response_mask, tokenizer)
    all_boundaries = []

    for b in range(batch_size):
        sentences = sentences_batch[b]
        nlp = neg_log_probs[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        think_end = think_boundaries[b]  # token index after </think>, or None

        # Compute mean -log pi of first 1/3 tokens for each sentence
        sent_scores = []
        for s_start, s_end in sentences:
            T = s_end - s_start
            head_len = max(1, T // 3)
            head_mean = nlp[s_start:s_start + head_len].mean().item()
            sent_scores.append(head_mean)

        # Split sentences into thinking vs answer sections
        if think_end is not None:
            think_sents = [(i, s) for i, s in enumerate(sentences) if s[0] < think_end]
            answer_sents = [(i, s) for i, s in enumerate(sentences) if s[0] >= think_end]
        else:
            think_sents = [(i, s) for i, s in enumerate(sentences)]
            answer_sents = []

        # Compute thresholds separately
        def _get_candidates(sent_list, pct):
            if len(sent_list) <= 1:
                return []
            scores = [sent_scores[i] for i, _ in sent_list]
            thresh = np.percentile(scores, pct)
            cands = []
            for idx_in_list, (sent_idx, (s_start, s_end)) in enumerate(sent_list):
                if idx_in_list == 0:
                    continue  # skip first sentence in each section
                if sent_scores[sent_idx] > thresh:
                    cands.append((sent_idx, sent_scores[sent_idx], s_start))
            return cands

        # Only split thinking section into phases; output = 1 phase
        candidates = _get_candidates(think_sents, percentile)

        # Debug: print sentence info for first response
        if b == 0:
            n_think = len(think_sents)
            n_answer = len(answer_sents)
            think_scores = [sent_scores[i] for i, _ in think_sents] if think_sents else []
            think_thresh = np.percentile(think_scores, percentile) if len(think_scores) > 1 else 0
            print(f"[ADPO Sentences] response=0: {len(sentences)} sentences "
                  f"(think={n_think}, output={n_answer}), "
                  f"think_end={think_end}, "
                  f"think_thresh={think_thresh:.4f} (p{percentile})",
                  flush=True)
            for i, (s_start, s_end) in enumerate(sentences[:8]):
                T = s_end - s_start
                section = "T" if think_end and s_start < think_end else "O"
                txt = tokenizer.decode(
                    token_ids[b, s_start:min(s_start+15, s_end)].tolist(),
                    skip_special_tokens=True,
                )
                is_cand = any(c[0] == i for c in candidates)
                mark = "✓" if is_cand else "✗"
                print(f"  sent {i} [{section}] (tok={s_start}, T={T}, score={sent_scores[i]:.4f}) "
                      f"{mark}: \"{txt[:80]}...\"", flush=True)

        # Build boundaries: thinking phases + 1 output phase
        candidates.sort(key=lambda x: -x[1])
        boundaries = [start]

        # Add thinking phase boundaries
        for _, _, s_start in candidates[:max_phases - 2]:  # reserve 1 slot for output phase
            if s_start not in boundaries:
                boundaries.append(s_start)

        # Always add </think> boundary (start of output phase)
        if think_end is not None:
            boundaries.append(think_end)

        boundaries.sort()
        all_boundaries.append(boundaries)

    return all_boundaries


def _detect_phase_boundaries_peak(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 85.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Legacy fallback: peak-based detection on -log pi."""
    batch_size = neg_log_probs.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        nlp = neg_log_probs[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        active_nlp = nlp[active].cpu().numpy()
        delta = np.percentile(active_nlp, percentile)

        candidates = []
        for t in range(start + 1, end - 1):
            val = nlp[t].item()
            if (val > delta
                    and val >= nlp[t - 1].item()
                    and val >= nlp[t + 1].item()):
                candidates.append((t, val))

        candidates.sort(key=lambda x: -x[1])
        boundaries = [start]
        for t, val in candidates:
            if len(boundaries) >= max_phases:
                break
            if all(abs(t - b_) >= min_phase_len for b_ in boundaries):
                boundaries.append(t)

        boundaries.sort()
        all_boundaries.append(boundaries)

    return all_boundaries


def compute_token_entropy(
    log_probs: torch.Tensor = None,
    logits: torch.Tensor = None,
    response_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-token entropy from logits or approximate from log_probs.

    If logits are provided, computes exact entropy:
        H(t) = -sum_v p(v|context) * log p(v|context)

    If only log_probs (of the chosen token) are available, uses -log_prob
    as an approximation (higher -log_prob ≈ higher uncertainty).

    Returns:
        entropy: (batch, seq_len) tensor of per-token entropy values.
    """
    if logits is not None:
        # Exact entropy from full vocabulary distribution
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1)  # (batch, seq_len)
    elif log_probs is not None:
        # Approximation: -log p(chosen token) as proxy for entropy
        # Higher -log_prob → model less certain → higher entropy
        entropy = -log_probs
    else:
        raise ValueError("Either logits or log_probs must be provided")

    if response_mask is not None:
        entropy = entropy * response_mask

    return entropy


def detect_phase_boundaries_entropy(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 80.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[List[int]]:
    """Detect phase boundaries using sentence structure + mean entropy.

    1. Split response into sentences by punctuation delimiters (. ? ! \\n ...).
    2. For each sentence, compute mean entropy of the first 1/3 tokens.
    3. If that mean > percentile threshold → sentence starts a new phase.
    4. Otherwise → sentence merges into the previous phase.

    Falls back to peak-based detection if token_ids/tokenizer not available.
    """
    if token_ids is None or tokenizer is None:
        # Fallback: treat entropy as signal for peak-based detection
        return _detect_phase_boundaries_peak(
            entropy, response_mask, percentile, min_phase_len, max_phases,
        )

    batch_size = entropy.shape[0]
    sentences_batch = _find_sentence_boundaries(token_ids, response_mask, tokenizer)
    all_boundaries = []

    for b in range(batch_size):
        sentences = sentences_batch[b]
        ent = entropy[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()

        # Compute mean entropy of first 1/3 tokens for each sentence
        sent_scores = []
        for s_start, s_end in sentences:
            T = s_end - s_start
            head_len = max(1, T // 3)
            head_mean = ent[s_start:s_start + head_len].mean().item()
            sent_scores.append(head_mean)

        threshold = np.percentile(sent_scores, percentile)

        # Rank all sentences (except first) by score, pick top-K above threshold
        candidates = []
        for i in range(1, len(sentences)):
            if sent_scores[i] > threshold:
                candidates.append((i, sent_scores[i], sentences[i][0]))

        candidates.sort(key=lambda x: -x[1])
        boundaries = [start]
        for _, _, s_start in candidates[:max_phases - 1]:
            boundaries.append(s_start)

        boundaries.sort()
        all_boundaries.append(boundaries)

    return all_boundaries


def detect_phase_boundaries(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    method: str = "adaptive",
    delta: float = 2.0,
    percentile: float = 85.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
    entropy: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[List[int]]:
    """Unified boundary detection dispatcher.

    Methods:
        threshold - Fixed -log pi threshold.
        adaptive  - Sentence-based + mean -log pi of first 1/3 tokens.
        entropy   - Sentence-based + mean entropy of first 1/3 tokens.
    """
    if method == "threshold":
        return detect_phase_boundaries_threshold(
            neg_log_probs, response_mask, delta=delta,
            min_phase_len=min_phase_len,
        )
    elif method == "adaptive":
        return detect_phase_boundaries_adaptive(
            neg_log_probs, response_mask, percentile=percentile,
            min_phase_len=min_phase_len, max_phases=max_phases,
            token_ids=token_ids, tokenizer=tokenizer,
        )
    elif method == "entropy":
        if entropy is None:
            entropy = neg_log_probs
        return detect_phase_boundaries_entropy(
            entropy, response_mask, percentile=percentile,
            min_phase_len=min_phase_len, max_phases=max_phases,
            token_ids=token_ids, tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown boundary method: {method}")


# ---------------------------------------------------------------------------
# Phase Segmentation
# ---------------------------------------------------------------------------

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
            text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)

        phases.append(PhaseSegment(
            phase_id=k, start_idx=start, end_idx=end, text=text,
        ))
    return phases


def build_phase_mask(
    boundaries_batch: List[List[int]],
    seq_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build tensor mapping each token to its phase index.

    Returns:
        phase_ids: (batch, seq_len) integer tensor.
                   -1 for non-response tokens.
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


# ---------------------------------------------------------------------------
# Phase-Level Advantage Computation
# ---------------------------------------------------------------------------

def compute_local_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LOCAL advantages: phase k vs other phases within the SAME response.

    A_{i,k}^local = r_{i,k} - mean(r_{i,1}, ..., r_{i,K})

    This tells us: "Is phase k better or worse than the average phase
    in this response?"

    Returns:
        local_advantages: (batch, max_K)
        local_stds: (batch,) std of phase rewards within each response.
    """
    batch_size, max_K = phase_rewards.shape

    # Mean reward per response (only over existing phases)
    phase_count = phase_mask.sum(dim=1).clamp(min=1)  # (batch,)
    mean_per_response = (phase_rewards * phase_mask).sum(dim=1) / phase_count  # (batch,)

    # Local advantage: r_{i,k} - mean_k(r_{i,k})
    local_advantages = (phase_rewards - mean_per_response.unsqueeze(1)) * phase_mask

    # Std per response (for adaptive lambda)
    # sigma_i^local = std of phase rewards within response i
    sq_diff = ((phase_rewards - mean_per_response.unsqueeze(1)) ** 2) * phase_mask
    variance = sq_diff.sum(dim=1) / phase_count.clamp(min=1)
    local_stds = torch.sqrt(variance + eps)  # (batch,)

    return local_advantages, local_stds


def compute_global_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GLOBAL advantages: response i vs other G responses (GRPO-style).

    Uses the MEAN of phase rewards as overall response score:
        score_i = mean(r_{i,1}, ..., r_{i,K})
        A_{i,k}^global = score_i - mean_j(score_j)

    This tells us: "Is this response overall better or worse than the
    group average?" Applied uniformly to all phases.

    NOTE: Does NOT divide by std (Dr.GRPO style) to avoid the
    small-std amplification problem.

    Returns:
        global_advantages: (batch, max_K) -- same value per phase within a response.
        global_std: (scalar per group) std of response scores across G generations.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    # Response-level score = mean of phase rewards
    phase_count = phase_mask.sum(dim=1).clamp(min=1)
    response_scores = (phase_rewards * phase_mask).sum(dim=1) / phase_count  # (batch,)

    global_advantages = torch.zeros_like(phase_rewards)
    # sigma^global: std of response scores across G generations (one scalar per group)
    # We store per-sample so each response knows its group's std
    global_stds = torch.zeros(batch_size, device=device)

    unique_indices = torch.unique(index)

    for uid in unique_indices:
        group_mask = (index == uid)
        group_scores = response_scores[group_mask]
        group_mean = group_scores.mean()
        group_std = group_scores.std() if group_scores.numel() > 1 else torch.tensor(0.0, device=device)

        # A_global = score_i - mean(scores) (Dr.GRPO: no division by std)
        adv = group_scores - group_mean
        # Broadcast same global advantage to all phases
        global_advantages[group_mask] = adv.unsqueeze(1) * phase_mask[group_mask]

        # sigma^global = std of response scores in this group
        global_stds[group_mask] = group_std

    return global_advantages, global_stds


def compute_phase_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ADPO advantages: adaptive weighted mean of local and global.

    A_{i,k} = lambda * A_{i,k}^local + (1 - lambda) * A_{i,k}^global

    where:
        lambda = sigma^global / (sigma^global + sigma_i^local + eps)

        sigma^global = std of response scores across G generations
        sigma_i^local = std of phase rewards within response i

    Intuition:
    - sigma^global HIGH (responses vary a lot across group)
      → lambda HIGH → trust local signal more (global is noisy)
    - sigma_i^local HIGH (phases vary a lot within response)
      → lambda LOW → trust global signal more (local is noisy)

    This avoids Dr.GRPO's problem: instead of dividing by std (which
    amplifies noise when std≈0), we adaptively weight between two
    complementary signals.
    """
    batch_size, max_K = phase_rewards.shape
    device = phase_rewards.device

    # Local: phase k vs other phases in same response
    local_adv, local_stds = compute_local_advantages(
        phase_rewards, phase_mask, eps=eps,
    )
    # local_stds: (batch,) -- sigma_i^local per response

    # Global: response i vs other responses (GRPO-style, no std division)
    global_adv, global_stds = compute_global_advantages(
        phase_rewards, phase_mask, index, eps=eps,
    )
    # global_stds: (batch,) -- sigma^global per response (same within a group)

    # Adaptive lambda = sigma^global / (sigma^global + sigma_i^local + eps)
    # Both are (batch,), broadcast to (batch, max_K) via unsqueeze
    lam = global_stds / (global_stds + local_stds + eps)  # (batch,)
    lam = lam.unsqueeze(1).expand_as(phase_rewards)       # (batch, max_K)

    # Adaptive weighted mean
    phase_advantages = lam * local_adv + (1 - lam) * global_adv
    phase_advantages = phase_advantages * phase_mask

    return phase_advantages


def assign_phase_advantages_to_tokens(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
    decay_gamma: float = 0.0,
    boundaries_batch: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Map phase-level advantages back to token-level.

    Hard: A(t) = A^(k(t))
    Decreasing (decay_gamma > 0): Within each phase, token advantages decay
        geometrically so earlier tokens get more credit:
            c = A_phase / (2 * T)
            a'_1 = A * (1 - gamma) * T / (9 * gamma)
            a_i = c + a'_i
            a'_{i+1} = a'_i * gamma
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device

    if decay_gamma > 0 and boundaries_batch is not None:
        token_advantages = _assign_with_decay(
            phase_advantages, phase_ids, response_mask,
            boundaries_batch, decay_gamma,
        )
    else:
        token_advantages = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            for t in range(seq_len):
                k = phase_ids[b, t].item()
                if k >= 0:
                    token_advantages[b, t] = phase_advantages[b, k]

    return token_advantages * response_mask


def _assign_with_decay(
    phase_advantages: torch.Tensor,
    phase_ids: torch.Tensor,
    response_mask: torch.Tensor,
    boundaries_batch: List[List[int]],
    gamma: float,
) -> torch.Tensor:
    """In-phase advantage decreasing: earlier tokens get more credit.

    For each phase with advantage A and T tokens:
        c       = A / (2 * T)       (constant baseline per token)
        a'_1    = A * (1 - gamma) * T / (9 * gamma)  (initial decay value)
        a_i     = c + a'_i          (token i gets baseline + decayed part)
        a'_{i+1}= a'_i * gamma      (geometric decay)
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device
    token_advantages = torch.zeros(batch_size, seq_len, device=device)

    logged = False
    for b in range(batch_size):
        boundaries = boundaries_batch[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            continue
        resp_end = active[-1].item() + 1

        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else resp_end
            T = end - start
            if T <= 0:
                continue

            A = phase_advantages[b, k].item()
            c = A / (2.0 * T)
            a_prime = A * (1 - gamma) * T / (9 * gamma)  # a'_1
            for i in range(T):
                token_advantages[b, start + i] = c + a_prime
                a_prime *= gamma

            # Log first response's decay info
            if not logged and T > 1:
                first_adv = token_advantages[b, start].item()
                last_adv = token_advantages[b, end - 1].item()
                print(
                    f"[ADPO Decay] b={b} phase={k} | T={T} A={A:.6e} "
                    f"gamma={gamma} | adv_first={first_adv:.6e} "
                    f"adv_last={last_adv:.6e} diff={first_adv - last_adv:.6e}",
                    flush=True,
                )
        if not logged:
            logged = True

    return token_advantages


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def compute_adpo_phase_advantages(
    log_probs: torch.Tensor,
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    boundaries_batch: List[List[int]],
    decay_gamma: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Full ADPO pipeline: phase rewards -> phase advantages -> token advantages.

    Called after LLM-as-Judge has scored each phase.

    Advantages are computed as adaptive weighted mean of:
    - Local: phase k vs other phases within same response
    - Global: response i vs other G responses (Dr.GRPO-style, no std division)

    lambda_k = sigma_k^global / (sigma_k^global + sigma_i^local + eps)
    A_{i,k} = lambda_k * A_local + (1-lambda_k) * A_global

    Token-level mapping modes:
    - decay_gamma > 0: In-phase decreasing — earlier tokens get more credit.
    - Otherwise: Hard assignment (all tokens in a phase share the same adv).
    """
    seq_len = response_mask.shape[1]

    phase_advantages = compute_phase_advantages(
        phase_rewards=phase_rewards,
        phase_mask=phase_mask,
        index=index,
        eps=eps,
    )

    phase_ids = build_phase_mask(boundaries_batch, seq_len, response_mask)

    token_advantages = assign_phase_advantages_to_tokens(
        phase_advantages=phase_advantages,
        phase_ids=phase_ids,
        response_mask=response_mask,
        decay_gamma=decay_gamma,
        boundaries_batch=boundaries_batch,
    )

    return token_advantages
