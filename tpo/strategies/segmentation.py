"""
Segmentation strategies for TPO.

A SegmentationStrategy controls how a reasoning trajectory is divided into
steps: it returns the vLLM generation parameters used when expanding a node.

The distinction between a *step* and a *leaf* is determined by the algorithm
(via :func:`~mc_analysis.vllm_helpers._is_terminal_node`):

* A node whose generation ends with a natural EOS or is truncated at
  ``max_tokens`` without a matching stop string is treated as a leaf.
* A node that ends with a configured stop string is a step boundary and will
  be expanded further (subject to the TerminationStrategy).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

Node = Dict[str, Any]


class SegmentationStrategy(ABC):
    """Abstract base for segmentation strategies.

    Implement :meth:`get_generation_params` to return the keyword arguments
    forwarded to vLLM's ``/completions`` endpoint for each node expansion.

    Required keys in the returned dict:
        ``max_tokens``  – int
        ``temperature`` – float
        ``top_p``       – float
    Optional key:
        ``stop``        – ``List[str]`` or ``None``
    """

    @abstractmethod
    def get_generation_params(self, node: Node, depth: int) -> Dict[str, Any]:
        """Return vLLM generation parameters for expanding *node* at *depth*."""


class FixedTokenSegmentation(SegmentationStrategy):
    """Each step generates up to *max_tokens* tokens — identical to SPO.

    When ``stop`` is ``None`` (default) the model generates until it hits the
    token budget or produces a natural EOS.  In this mode every child is
    immediately a leaf (SPO's "M-token" mode).

    When ``stop`` is provided, the model generates until it hits a stop string
    or the token budget; nodes ending with a stop string are expanded further.

    Parameters
    ----------
    max_tokens : int
        Token budget per step.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    stop : list of str, optional
        Intermediate stop sequences that mark step boundaries.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def get_generation_params(self, node: Node, depth: int) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
        }


class StopSequenceSegmentation(SegmentationStrategy):
    """Segment reasoning at explicit stop strings (e.g. step markers or newlines).

    The model generates until it produces one of the *stop* strings or hits the
    *max_tokens* budget.  Nodes ending with a stop string are step boundaries
    and will be expanded further; nodes that hit the budget or a natural EOS
    are leaves.

    Parameters
    ----------
    stop : list of str
        Non-empty list of stop strings that delimit reasoning steps.
    max_tokens : int
        Token budget per step.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    """

    def __init__(
        self,
        stop: List[str],
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
    ):
        if not stop:
            raise ValueError("stop must be a non-empty list of strings")
        self.stop = stop
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def get_generation_params(self, node: Node, depth: int) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
        }


class AdaptiveTokenSegmentation(SegmentationStrategy):
    """Token budget that shrinks geometrically with depth.

    Allocates more tokens to early (shallower) steps and fewer to later ones.
    This reflects the intuition that early reasoning steps often require more
    exploration while later steps are more constrained.

    Budget at depth *d*:
        ``max(min_max_tokens, floor(initial_max_tokens × decay_per_depth^d))``

    Parameters
    ----------
    initial_max_tokens : int
        Token budget at depth 0.
    min_max_tokens : int
        Floor on the token budget (never goes below this).
    decay_per_depth : float
        Multiplicative decay per depth level (0 < decay_per_depth ≤ 1).
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    stop : list of str, optional
        Intermediate stop sequences.
    """

    def __init__(
        self,
        initial_max_tokens: int = 1024,
        min_max_tokens: int = 128,
        decay_per_depth: float = 0.7,
        temperature: float = 0.8,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ):
        if not (0.0 < decay_per_depth <= 1.0):
            raise ValueError("decay_per_depth must be in (0, 1]")
        self.initial_max_tokens = initial_max_tokens
        self.min_max_tokens = min_max_tokens
        self.decay_per_depth = decay_per_depth
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def get_generation_params(self, node: Node, depth: int) -> Dict[str, Any]:
        budget = max(
            self.min_max_tokens,
            int(self.initial_max_tokens * (self.decay_per_depth ** depth)),
        )
        return {
            "max_tokens": budget,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
        }


class DelayBranchingSegmentation(SegmentationStrategy):
    """Delay branching: generate a long chunk, then truncate at a high-uncertainty position.

    **Algorithm** (applied per child after generation):

    1. Compute a forward-looking average uncertainty for each token position *i*::

           avg_uncertainty[i] = mean(-logprob[j] for j in [i, i+1, …, i+window_size-1])

       Using the selected token's negative log-prob as a per-token uncertainty proxy.
       Positions near the end (where the window extends past the sequence) are
       computed over the available tokens only.

    2. The segmentation threshold is the *entropy_percentile*-th percentile of
       all ``avg_uncertainty`` values in the current chunk (computed per-child,
       so the threshold adapts to each sampled sequence).

    3. The **first** position *i* ≥ *min_segment_tokens* where
       ``avg_uncertainty[i] ≥ threshold`` is chosen as the cut point.

    4. The child's text is truncated to ``tokens[:cut_idx]`` and its metadata
       (``full_text``, ``tokens``, ``token_logprobs``, ``sum_logprobs``,
       ``num_tokens``, ``finish_reason``) are updated accordingly by the
       algorithm.  The truncated node is then treated as a step boundary and
       expanded further (unless a termination strategy stops it).

    If no position satisfies the criterion, the child is returned unchanged —
    it becomes a leaf in the usual way.

    .. note::
        This strategy requires per-token log-probs, so ``get_logprobs=True``
        must be passed to :func:`~tpo.algorithm.build_tree_tpo`.  Call
        :meth:`needs_per_token_logprobs` to verify at runtime.

    Parameters
    ----------
    max_tokens : int
        Token budget for the initial (pre-truncation) generation.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    stop : list of str, optional
        Stop sequences forwarded to vLLM unchanged.
    window_size : int
        Look-ahead window K for the sliding average.
    entropy_percentile : float
        Percentile threshold ∈ (0, 100].  Positions above this percentile
        of per-child average uncertainties are candidate cut points.
    min_segment_tokens : int
        Minimum number of tokens in a segment (suppresses cutting too early).
    """

    def __init__(
        self,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        window_size: int = 5,
        entropy_percentile: float = 80.0,
        min_segment_tokens: int = 10,
    ):
        if not (0.0 < entropy_percentile <= 100.0):
            raise ValueError("entropy_percentile must be in (0, 100]")
        if window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        if min_segment_tokens < 0:
            raise ValueError("min_segment_tokens must be ≥ 0")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop
        self.window_size = window_size
        self.entropy_percentile = entropy_percentile
        self.min_segment_tokens = min_segment_tokens

    def needs_per_token_logprobs(self) -> bool:
        """Signal that the algorithm must request per-token log-probs (get_logprobs=True)."""
        return True

    def get_generation_params(self, node: Node, depth: int) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
        }

    def find_segment_point(self, token_logprobs: List[float]) -> Optional[int]:
        """Return the index at which to cut the token sequence, or ``None``.

        Parameters
        ----------
        token_logprobs : list of float
            Per-token log-probabilities (negative values).

        Returns
        -------
        int or None
            Cut index (exclusive upper bound for ``tokens[:cut_idx]``), or
            ``None`` if no segmentation point is found.
        """
        import numpy as np

        n = len(token_logprobs)
        if n == 0 or n <= self.min_segment_tokens:
            return None

        lp = np.array(token_logprobs, dtype=np.float64)
        uncertainty = -lp  # negative log-prob → higher = more uncertain

        # Forward-looking sliding average
        avg = np.empty(n, dtype=np.float64)
        for i in range(n):
            end = min(i + self.window_size, n)
            avg[i] = uncertainty[i:end].mean()

        threshold = float(np.percentile(avg, self.entropy_percentile))

        # First position ≥ min_segment_tokens where avg exceeds threshold
        for i in range(self.min_segment_tokens, n):
            if avg[i] >= threshold:
                return i

        return None
