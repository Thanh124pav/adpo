import torch
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .base import BaseReward
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)

class EntropyReward(BaseReward):
    def __init__(self, config):
        super().__init__()
        algo = config.algorithm
        self.psi = getattr(algo, 'psi', 0.95*0.99)
        self.threshold_pct = getattr(algo, 'default_threshold_percentile', 90.0)
        self.correct_total = getattr(algo,'correct_total', 1.0)
        self.incorrect_total = getattr(algo, 'incorrect_total', -1.0)
        self.partial_total = getattr(algo, 'partial_total', 0.1)

    def compute(self, boundaries_batch, response_mask, index, entropy, outcome_rewards, **context) \
        -> Tuple[torch.Tensor, torch.Tensor, dict]:
        batch_size, seq_len = response_mask.shape
        all_cum_entropy = compute_phase_cumulative_entropy(entropy, response_mask, boundaries_batch, self.psi)
        phase_rewards_batch = compute_entropy_credit_rewards(all_cum_entropy, outcome_rewards, index, self.correct_total, self.incorrect_total, self.partial_total, self.threshold_pct)

        max_K = max(len(b) for b in boundaries_batch)
        phase_rewards_tensor = torch.zeros(batch_size, max_K, device=response_mask.device)
        phase_mask_tensor = torch.zeros(batch_size, max_K, device=response_mask.device)
        for b in range(batch_size):
            n_phases = len(phase_rewards_batch[b])
            for k in range(n_phases):
                phase_rewards_tensor[b, k] = float(phase_rewards_batch[b][k])
                phase_mask_tensor[b,k] = 1.0
        return phase_rewards_tensor, phase_mask_tensor, {}


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