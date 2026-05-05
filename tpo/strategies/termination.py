"""
Termination strategies for TPO.

A TerminationStrategy decides whether a node should be treated as a leaf
(not expanded further), based on the full path from the root to that node.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

Node = Dict[str, Any]


class TerminationStrategy(ABC):
    """Abstract base for node-termination strategies.

    :meth:`should_terminate` is called for every freshly sampled child node
    *before* deciding whether to expand it.  It receives the complete path
    from root to the candidate node (inclusive) so that strategies can
    inspect ancestry.

    If the method returns ``True``, the node is treated as a leaf: an answer
    is extracted from its text and the DFS stops.  Natural EOS / stop-string
    termination is handled separately by the algorithm and is *not* routed
    through this method.
    """

    @abstractmethod
    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        """Return ``True`` if *node* should not be expanded further.

        Parameters
        ----------
        node : Node
            The candidate node (deepest element of *path*).
        path : List[Node]
            Full path ``[root, …, parent, node]`` from root to *node*,
            inclusive.  Every node in the path has already been sampled
            and annotated with at least ``"depth"`` and, when relevant,
            ``"P"`` (log P(answer|trajectory)).
        """


class DepthTerminationStrategy(TerminationStrategy):
    """Terminate when a node's depth reaches *max_depth* — same as SPO.

    Parameters
    ----------
    max_depth : int
        Nodes at this depth (or deeper) are not expanded further.
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        return node.get("depth", len(path) - 1) >= self.max_depth


class PDegradationTerminationStrategy(TerminationStrategy):
    """Terminate when P(answer|trajectory) degrades too many times.

    **Core idea** (TPO spec):

    For the path ``[n_0 = root, n_1, …, n_k = node]``, count the number of
    nodes ``n_i`` (i ≥ 1) where

        P(answer | trajectory to n_i) < P(answer | trajectory to parent n_{i-1})

    i.e. where the model's confidence in the gold answer *decreased* after
    incorporating the reasoning step that produced ``n_i``.

    If this **degradation count exceeds K** (*max_degradations*), the node is
    treated as a terminal leaf — further reasoning along this branch is
    unlikely to reach the correct answer.

    A hard *max_depth* cap is also enforced as a safety backstop.

    .. note::
        ``node["P"]`` (log P of the gold answer given the trajectory up to
        that node) must be populated for every node in *path* before this
        strategy is called.  :func:`~tpo.algorithm.build_tree_tpo` handles
        this automatically by computing P inline for each sampled node.

    Parameters
    ----------
    max_degradations : int
        Threshold K.  A node is terminal when the degradation count
        **strictly exceeds** K (i.e. count > K).
    max_depth : int
        Hard depth cap applied independently of P degradation.
    """

    def __init__(self, max_degradations: int = 3, max_depth: int = 10):
        if max_degradations < 0:
            raise ValueError("max_degradations must be ≥ 0")
        self.K = max_degradations
        self.max_depth = max_depth

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        if node.get("depth", len(path) - 1) >= self.max_depth:
            return True

        # Count P degradations along path (skip root at index 0)
        degradations = 0
        for i in range(1, len(path)):
            p_curr = path[i].get("P")
            p_par = path[i - 1].get("P")
            if p_curr is not None and p_par is not None and p_curr < p_par:
                degradations += 1

        return degradations > self.K

    def degradation_count(self, path: List[Node]) -> int:
        """Utility: return the current degradation count for a given path."""
        count = 0
        for i in range(1, len(path)):
            p_curr = path[i].get("P")
            p_par = path[i - 1].get("P")
            if p_curr is not None and p_par is not None and p_curr < p_par:
                count += 1
        return count


class CombinedTerminationStrategy(TerminationStrategy):
    """Terminate when **any** of the supplied strategies says to terminate.

    Useful for layering multiple stopping conditions, e.g. combining a depth
    cap with P-degradation:

    .. code-block:: python

        strategy = CombinedTerminationStrategy([
            DepthTerminationStrategy(max_depth=6),
            PDegradationTerminationStrategy(max_degradations=2),
        ])

    Parameters
    ----------
    strategies : list of TerminationStrategy
        The strategies to consult.  The first one that returns ``True``
        short-circuits evaluation.
    """

    def __init__(self, strategies: List[TerminationStrategy]):
        if not strategies:
            raise ValueError("strategies must be non-empty")
        self.strategies = strategies

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        return any(s.should_terminate(node, path) for s in self.strategies)
