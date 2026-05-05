"""
Branching strategies for TPO.

A BranchingStrategy decides how many children to sample at each node.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

Node = Dict[str, Any]

# Hard limits mandated by the TPO specification.
BRANCH_MIN = 2
BRANCH_MAX = 8


class BranchingStrategy(ABC):
    """Abstract base for branching strategies.

    Implement :meth:`get_branch_count` to control how many children are
    sampled per node.  If your strategy needs the node's next-token
    distribution to be pre-fetched, override :meth:`needs_top_logprobs`
    to return ``True``; the algorithm will then populate
    ``node["top_logprobs"]`` before calling :meth:`get_branch_count`.
    """

    @abstractmethod
    def get_branch_count(self, node: Node, depth: int) -> int:
        """Return the number of children to sample from *node* at *depth*."""

    def needs_top_logprobs(self) -> bool:
        """Override to ``True`` if :meth:`get_branch_count` reads ``node["top_logprobs"]``."""
        return False


class FixedBranchingStrategy(BranchingStrategy):
    """Constant (or depth-indexed) branch factor — identical to SPO's behaviour.

    Parameters
    ----------
    branch_factor : int or List[int]
        If an ``int``, every node is expanded with that many children.
        If a list, ``branch_factor[depth]`` is used; the last element is
        reused for any depth beyond the list length.
    """

    def __init__(self, branch_factor: Union[int, List[int]] = 3):
        self.branch_factor = branch_factor

    def get_branch_count(self, node: Node, depth: int) -> int:
        if isinstance(self.branch_factor, int):
            return self.branch_factor
        idx = min(depth, len(self.branch_factor) - 1)
        return self.branch_factor[idx]


class EntropyBranchingStrategy(BranchingStrategy):
    """Branch count driven by the entropy of the next-token distribution.

    The model's uncertainty at the current node determines how many
    reasoning paths are worth exploring:

    * **High entropy** (model uncertain about next token) → many branches
      (up to *max_branches*, capped at 8).
    * **Low entropy** (model is confident) → few branches
      (down to *min_branches*, floored at 2).

    Entropy is estimated from the top-*top_k* next-token log-probs stored in
    ``node["top_logprobs"]`` (a list of ``(token, log_prob)`` pairs).  The
    algorithm populates this field automatically because :meth:`needs_top_logprobs`
    returns ``True``.

    **Normalisation**:  Probabilities are re-normalised to sum to 1 over the
    top-K tokens (conditioned view), then Shannon entropy is computed and
    divided by ``log(top_k)`` — the maximum possible entropy for *top_k*
    equally-probable tokens.  The resulting value in ``[0, 1]`` is linearly
    mapped to ``[min_branches, max_branches]``.

    Parameters
    ----------
    min_branches : int
        Minimum branch count (≥ 2, per TPO spec).
    max_branches : int
        Maximum branch count (≤ 8, per TPO spec).
    top_k : int
        Number of top tokens used for entropy estimation.  Should match the
        ``top_k_entropy`` parameter passed to :func:`~tpo.algorithm.build_tree_tpo`.
    """

    def __init__(
        self,
        min_branches: int = BRANCH_MIN,
        max_branches: int = BRANCH_MAX,
        top_k: int = 20,
    ):
        if min_branches < BRANCH_MIN:
            raise ValueError(f"min_branches must be ≥ {BRANCH_MIN}, got {min_branches}")
        if max_branches > BRANCH_MAX:
            raise ValueError(f"max_branches must be ≤ {BRANCH_MAX}, got {max_branches}")
        if min_branches > max_branches:
            raise ValueError("min_branches must be ≤ max_branches")
        self.min_branches = min_branches
        self.max_branches = max_branches
        self.top_k = top_k

    def needs_top_logprobs(self) -> bool:
        return True

    def get_branch_count(self, node: Node, depth: int) -> int:
        top_logprobs: Optional[List[Tuple[str, float]]] = node.get("top_logprobs")
        if not top_logprobs:
            # Fallback when logprobs are unavailable
            return self.min_branches

        h_norm = _normalized_entropy(top_logprobs)
        span = self.max_branches - self.min_branches
        raw = self.min_branches + h_norm * span
        return max(self.min_branches, min(self.max_branches, round(raw)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalized_entropy(top_logprobs: List[Tuple[str, float]]) -> float:
    """Normalised Shannon entropy ∈ [0, 1] from top-K (token, log_prob) pairs.

    Probabilities are renormalised over the top-K tokens so that their sum
    equals 1.  Entropy is then divided by ``log(K)`` (maximum for a uniform
    distribution over K tokens), giving a value in ``[0, 1]``.

    This deliberately ignores residual probability mass outside the top-K set:
    the renormalisation makes the estimate a *lower bound* on true entropy,
    which is conservative and appropriate for deciding branching width.
    """
    if not top_logprobs:
        return 0.0

    probs = np.array([math.exp(lp) for _, lp in top_logprobs], dtype=np.float64)
    total = probs.sum()
    if total < 1e-12:
        return 0.0
    probs /= total  # renormalise to sum=1

    mask = probs > 1e-12
    H = -float(np.sum(probs[mask] * np.log(probs[mask])))
    H_max = math.log(len(probs))  # log(K)

    if H_max < 1e-12:
        return 0.0
    return min(H / H_max, 1.0)
