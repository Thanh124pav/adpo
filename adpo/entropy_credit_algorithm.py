"""
Entropy-Credit Algorithm: Phase decomposition using sliding-window entropy,
credit assignment using cumulative entropy (no attention reconstruction).

Core idea:
1. Segment response into phases using sliding-window entropy (same as pure-entropy).
2. Last phase reward = exact matching score.
3. For phases 0..m-2: compute cumulative entropy with decay factor psi:
       E_phase = e_1 + e_2*psi + e_3*psi^2 + ... + e_T*psi^(T-1)
4. Determine threshold P from correct responses in the group.
   If all responses are wrong, use percentile from config.
5. Phase reward = linear function of distance from threshold:
       R = min(1, (P - E) / (P_100 - P))
   where P_100 = max cumulative entropy across all phases.
6. Advantages: local + global (same as ADPO), with in-phase decay gamma.

No LLM-as-Judge, no attention reconstruction, no HF model needed.
Much faster than pure-entropy since it only uses entropy from rollout.
"""

import math
import logging
import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase Boundary Detection (reused from pure_entropy_algorithm)
# ---------------------------------------------------------------------------

def detect_phase_boundaries_entropy_credit(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    window_size: int = 10,
    percentile: float = 75.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Detect phase boundaries using sliding-window entropy percentile.

    Same algorithm as pure_entropy_algorithm.detect_phase_boundaries_pure_entropy.
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

        ent_vals = ent[start:end].float().cpu().numpy()
        n_windows = resp_len - window_size + 1
        window_means = np.array([
            ent_vals[i:i + window_size].mean()
            for i in range(n_windows)
        ])

        threshold = np.percentile(window_means, percentile)

        candidates = []
        for i in range(n_windows):
            if window_means[i] > threshold:
                candidates.append((start + i, window_means[i]))

        boundaries = [start]
        candidates.sort(key=lambda x: -x[1])
        for pos, score in candidates:
            if len(boundaries) >= max_phases:
                break
            if all(abs(pos - bd) >= min_phase_len for bd in boundaries):
                boundaries.append(pos)

        boundaries.sort()
        all_boundaries.append(boundaries)

        if b == 0:
            print(
                f"[EntropyCredit Boundaries] resp=0: resp_len={resp_len}, "
                f"window_size={window_size}, threshold={threshold:.4f}, "
                f"n_candidates={len(candidates)}, "
                f"n_phases={len(boundaries)}, boundaries={boundaries}",
                flush=True,
            )

    return all_boundaries


# ---------------------------------------------------------------------------
# Cumulative Entropy per Phase
# ---------------------------------------------------------------------------

def compute_phase_cumulative_entropy(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    boundaries_batch: List[List[int]],
    psi: float = 0.95,
) -> List[List[float]]:
    """Compute cumulative entropy for each phase with decay factor psi.

    For a phase with T tokens having entropy [e_1, e_2, ..., e_T]:
        E_phase = e_1 + e_2*psi + e_3*psi^2 + ... + e_T*psi^(T-1)

    This gives more weight to earlier tokens in the phase (where high
    entropy signals uncertainty at the start of a reasoning step).

    Args:
        entropy: (batch, seq_len) per-token entropy.
        response_mask: (batch, seq_len) response mask.
        boundaries_batch: List of boundary lists per response.
        psi: Decay factor (default 0.95). psi=1.0 means no decay.

    Returns:
        List of lists: cum_entropy[b][k] = cumulative entropy of phase k
        in response b.
    """
    batch_size = entropy.shape[0]
    all_cum_entropy = []

    for b in range(batch_size):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_cum_entropy.append([0.0] * len(boundaries_batch[b]))
            continue

        resp_end = active[-1].item() + 1
        ent = entropy[b].float().cpu().numpy()
        boundaries = boundaries_batch[b]

        phase_entropies = []
        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else resp_end
            T = end - start

            if T <= 0:
                phase_entropies.append(0.0)
                continue

            # e_1 + e_2*psi + e_3*psi^2 + ...
            cum_e = 0.0
            psi_power = 1.0
            for t in range(start, end):
                cum_e += ent[t] * psi_power
                psi_power *= psi

            phase_entropies.append(cum_e)

        all_cum_entropy.append(phase_entropies)

    return all_cum_entropy


# ---------------------------------------------------------------------------
# Entropy-Based Reward Assignment
# ---------------------------------------------------------------------------

def compute_entropy_credit_rewards(
    cum_entropy_batch: List[List[float]],
    outcome_rewards: List[float],
    index: torch.Tensor,
    correct_total: float = 1.0,
    incorrect_total: float = -1.0,
    partial_total: float = 0.1,
    default_percentile: float = 90.0,
) -> List[np.ndarray]:
    """Compute phase rewards based on cumulative entropy thresholds.

    Key constraint: sum(r_1, r_2, ..., r_m) = R_total for each response.
        R_total = correct_total (1.0) if answer is correct
        R_total = incorrect_total (-1.0) if answer is wrong
        R_total = partial_total (0.1) if no boxed answer / partial

    The DISTRIBUTION of R_total across phases is determined by entropy credit:
    1. Compute raw score per phase: raw[k] = min(1, (P - E_k) / (P_100 - P))
       - E_k < P → raw > 0 (good phase, low entropy)
       - E_k > P → raw < 0 (bad phase, high entropy)
    2. Shift all raw scores by a constant so their sum = R_total:
       r[k] = raw[k] + (R_total/m - mean(raw))

    Threshold P:
        - If group has correct responses: P = max cum_entropy of correct responses.
        - If all wrong: P = percentile(all_cum_entropies, default_percentile).

    Args:
        cum_entropy_batch: List of lists of cumulative entropies per phase.
        outcome_rewards: List of outcome scores (1.0 = correct, 0.0 = wrong).
        index: (batch,) group indices.
        correct_total: Total reward budget when answer is correct.
        incorrect_total: Total reward budget when answer is wrong.
        partial_total: Total reward budget when partial/no answer.
        default_percentile: Percentile for threshold when all responses wrong.

    Returns:
        List of np.ndarray: rewards per phase per response. sum = R_total.
    """
    batch_size = len(cum_entropy_batch)

    # Group responses by uid
    unique_uids = torch.unique(index).tolist()
    uid_to_indices = {}
    for b in range(batch_size):
        uid = index[b].item()
        if uid not in uid_to_indices:
            uid_to_indices[uid] = []
        uid_to_indices[uid].append(b)

    phase_rewards_batch = [None] * batch_size

    for uid, group_indices in uid_to_indices.items():
        # Collect all cumulative entropies in this group
        all_cum_e = []
        correct_cum_e = []
        for b in group_indices:
            for e in cum_entropy_batch[b]:
                all_cum_e.append(e)
            if outcome_rewards[b] >= 1.0:
                for e in cum_entropy_batch[b]:
                    correct_cum_e.append(e)

        if len(all_cum_e) == 0:
            for b in group_indices:
                n_phases = len(cum_entropy_batch[b])
                phase_rewards_batch[b] = np.zeros(n_phases)
            continue

        # P_100: max cumulative entropy across all phases in group
        P_100 = max(all_cum_e)

        # Determine threshold P
        has_correct = len(correct_cum_e) > 0
        if has_correct:
            P = max(correct_cum_e)
        else:
            P = np.percentile(all_cum_e, default_percentile)

        denom = P_100 - P + 1e-8

        # Log for first group
        if uid == unique_uids[0]:
            n_correct = sum(1 for b in group_indices if outcome_rewards[b] >= 1.0)
            print(
                f"[EntropyCredit Threshold] group uid={uid}: "
                f"{n_correct}/{len(group_indices)} correct, "
                f"P={P:.4f} ({'from correct' if has_correct else f'pct={default_percentile}'}), "
                f"P_100={P_100:.4f}, denom={denom:.4f}",
                flush=True,
            )

        # Compute rewards for each response in group
        for b in group_indices:
            cum_e = cum_entropy_batch[b]
            n_phases = len(cum_e)

            # Determine R_total for this response
            if outcome_rewards[b] >= 1.0:
                R_total = correct_total
            elif outcome_rewards[b] > 0.0:
                R_total = partial_total
            else:
                R_total = incorrect_total

            if n_phases <= 1:
                phase_rewards_batch[b] = np.array([R_total])
                continue

            # Raw credit per phase: min(1, (P - E_k) / (P_100 - P))
            raw = np.array([
                min(1.0, (P - cum_e[k]) / denom)
                for k in range(n_phases)
            ])

            # Shift so sum(rewards) = R_total
            # rewards[k] = raw[k] + shift, where shift = R_total/m - mean(raw)
            mean_raw = raw.mean()
            shift = R_total / n_phases - mean_raw
            rewards = raw + shift

            phase_rewards_batch[b] = rewards

            # Log first response of first group
            if b == group_indices[0] and uid == unique_uids[0]:
                print(
                    f"[EntropyCredit Rewards] resp={b}: R_total={R_total:.2f}, "
                    f"cum_entropy={[f'{e:.2f}' for e in cum_e]}",
                    flush=True,
                )
                print(
                    f"  raw_credit={[f'{r:.4f}' for r in raw]}, "
                    f"mean_raw={mean_raw:.4f}, shift={shift:.4f}",
                    flush=True,
                )
                print(
                    f"  rewards={[f'{r:.4f}' for r in rewards]}, "
                    f"sum={rewards.sum():.4f} (should={R_total:.2f})",
                    flush=True,
                )

    return phase_rewards_batch


# ---------------------------------------------------------------------------
# Token Entropy (reused)
# ---------------------------------------------------------------------------

def compute_token_entropy(
    log_probs: torch.Tensor = None,
    logits: torch.Tensor = None,
    response_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-token entropy from logits or approximate from log_probs."""
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
# Phase Segmentation
# ---------------------------------------------------------------------------

@dataclass
class PhaseSegment:
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
        phases.append(PhaseSegment(phase_id=k, start_idx=start, end_idx=end, text=text))
    return phases


# ---------------------------------------------------------------------------
# Phase Mask & Advantage Computation
# ---------------------------------------------------------------------------

def build_phase_mask(
    boundaries_batch: List[List[int]],
    seq_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build tensor mapping each token to its phase index. -1 for non-response."""
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


def compute_phase_advantages(
    phase_rewards: torch.Tensor,
    phase_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute advantages: local (within response) + global (across group).

    Same as ADPO: additive combination of local and global advantages.
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
    decay_gamma: float = 0.0,
    boundaries_batch: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Map phase-level advantages to tokens.

    If decay_gamma > 0: in-phase decreasing (earlier tokens get more credit).
    Otherwise: hard assignment (all tokens same advantage).
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
        c       = A / 3
        a'_1    = A / 2
        a_i     = c + a'_i
        a'_{i+1}= a'_i * gamma
    """
    batch_size, seq_len = phase_ids.shape
    device = phase_ids.device
    token_advantages = torch.zeros(batch_size, seq_len, device=device)

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
            c = A / 3.0
            a_prime = A / 2.0
            for i in range(T):
                token_advantages[b, start + i] = c + a_prime
                a_prime *= gamma

    return token_advantages


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def compute_entropy_credit_advantages(
    response_mask: torch.Tensor,
    index: torch.Tensor,
    boundaries_batch: List[List[int]],
    phase_rewards_batch: List[np.ndarray],
    decay_gamma: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full pipeline: phase rewards -> phase advantages -> token advantages.

    Returns:
        token_advantages: (batch, seq_len)
        phase_advantages: (batch, max_K)
        phase_rewards_tensor: (batch, max_K)
        phase_mask_tensor: (batch, max_K)
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

    # Phase advantages (local + global)
    phase_advantages = compute_phase_advantages(
        phase_rewards=phase_rewards,
        phase_mask=phase_mask_tensor,
        index=index,
        eps=eps,
    )

    # Phase IDs per token
    phase_ids = build_phase_mask(boundaries_batch, seq_len, response_mask)

    # Token advantages (with optional decay)
    token_advantages = assign_phase_advantages_to_tokens(
        phase_advantages=phase_advantages,
        phase_ids=phase_ids,
        response_mask=response_mask,
        decay_gamma=decay_gamma,
        boundaries_batch=boundaries_batch,
    )

    return token_advantages, phase_advantages, phase_rewards, phase_mask_tensor
