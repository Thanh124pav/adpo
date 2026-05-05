"""
TPO — Tree Path Optimization
============================

A search-tree algorithm derived from SPO with two core innovations:

1. **Entropy-driven dynamic branching** — the branch count at each node is
   determined by the Shannon entropy of the next-token distribution:
   high entropy → up to 8 branches; low entropy → as few as 2 branches.

2. **P-degradation termination** — a node is treated as a leaf when, along
   the root-to-node trajectory, the count of nodes whose
   ``P(answer | trajectory)`` is lower than their parent's exceeds a
   threshold K (hyperparameter).

The algorithm is fully extensible: swap in any custom
:class:`~tpo.strategies.BranchingStrategy`,
:class:`~tpo.strategies.TerminationStrategy`, or
:class:`~tpo.strategies.SegmentationStrategy` without touching the core loop.

Quick start
-----------
.. code-block:: python

    from tpo import analyse_tpo
    from tpo.strategies import (
        EntropyBranchingStrategy,
        PDegradationTerminationStrategy,
        FixedTokenSegmentation,
    )

    root = analyse_tpo(
        question      = "What is 17 × 23?",
        gold_answer   = "391",
        server_url    = "http://localhost:8000/v1",
        model_name    = "my-model",
        branching_strategy   = EntropyBranchingStrategy(min_branches=2, max_branches=8),
        termination_strategy = PDegradationTerminationStrategy(max_degradations=3, max_depth=8),
        segmentation_strategy= FixedTokenSegmentation(max_tokens=512, temperature=0.8),
    )

    # root is compatible with mc_analysis visualisation tools
    from mc_analysis import print_tree, visualise_html
    print_tree(root)

Public API
----------
"""

from .pipeline import analyse_tpo, analyse_tpo_async
from .algorithm import build_tree_tpo, build_tree_tpo_async
from .strategies import (
    # branching
    BranchingStrategy,
    FixedBranchingStrategy,
    EntropyBranchingStrategy,
    BRANCH_MIN,
    BRANCH_MAX,
    # termination
    TerminationStrategy,
    DepthTerminationStrategy,
    PDegradationTerminationStrategy,
    CombinedTerminationStrategy,
    # segmentation
    SegmentationStrategy,
    FixedTokenSegmentation,
    StopSequenceSegmentation,
    AdaptiveTokenSegmentation,
)

__all__ = [
    # pipeline
    "analyse_tpo",
    "analyse_tpo_async",
    # algorithm
    "build_tree_tpo",
    "build_tree_tpo_async",
    # branching
    "BranchingStrategy",
    "FixedBranchingStrategy",
    "EntropyBranchingStrategy",
    "BRANCH_MIN",
    "BRANCH_MAX",
    # termination
    "TerminationStrategy",
    "DepthTerminationStrategy",
    "PDegradationTerminationStrategy",
    "CombinedTerminationStrategy",
    # segmentation
    "SegmentationStrategy",
    "FixedTokenSegmentation",
    "StopSequenceSegmentation",
    "AdaptiveTokenSegmentation",
]
