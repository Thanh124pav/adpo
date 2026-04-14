import torch
import re
import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from .base import PhaseSplitter

logger = logging.getLogger(__name__)

class PureEntropySplitter(PhaseSplitter):
    def __init__(self, config):
        super(self, PureEntropySplitter).__init__()
        self.window_size = config.entropy_window_size
        self.percentile = config.entropy_percentile
        self.min_phase_len = config.phase_min_len
        self.max_phases = config.phase_max_K
        
    def split(
        self,
        entropy: torch.Tensor,
        response_mask: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        tokenizer=None,
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

            if resp_len <= self.window_size:
                all_boundaries.append([start])
                continue

            # Compute sliding-window means
            ent_vals = ent[start:end].float().cpu().numpy()
            n_windows = resp_len - self.window_size + 1
            window_means = np.array([
                ent_vals[i:i + self.window_size].mean()
                for i in range(n_windows)
            ])

            # Percentile threshold
            threshold = np.percentile(window_means, self.percentile)

            # Find candidate positions (in absolute token indices)
            candidates = []
            for i in range(n_windows):
                if window_means[i] > threshold:
                    candidates.append((start + i, window_means[i]))

            # Greedy selection respecting min_phase_len
            boundaries = [start]
            candidates.sort(key=lambda x: -x[1])  # highest first
            for pos, score in candidates:
                if len(boundaries) >= self.max_phases:
                    break
                if all(abs(pos - bd) >= self.min_phase_len for bd in boundaries):
                    boundaries.append(pos)

            boundaries.sort()
            all_boundaries.append(boundaries)

            # Log first response
            if b == 0:
                logger.info(
                    f"[PureEntropy Boundaries] resp=0: resp_len={resp_len}, "
                    f"window_size={self.window_size}, threshold={threshold:.4f}, "
                    f"n_candidates={len(candidates)}, "
                    f"n_phases={len(boundaries)}, boundaries={boundaries}"
                )
                print(
                    f"[PureEntropy Boundaries] resp=0: resp_len={resp_len}, "
                    f"window_size={self.window_size}, threshold={threshold:.4f}, "
                    f"n_candidates={len(candidates)}, "
                    f"n_phases={len(boundaries)}, boundaries={boundaries}",
                    flush=True,
                )

        return all_boundaries