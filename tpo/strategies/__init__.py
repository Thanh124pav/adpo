"""
TPO strategy plug-ins.

Branching
---------
BranchingStrategy         – abstract base
FixedBranchingStrategy    – constant branch factor (SPO-compatible)
EntropyBranchingStrategy  – dynamic branching from next-token entropy [2, 8]

Termination
-----------
TerminationStrategy            – abstract base
DepthTerminationStrategy       – max-depth cap (SPO-compatible)
PDegradationTerminationStrategy – terminates when P(answer|traj) degrades K times
CombinedTerminationStrategy    – OR of multiple strategies

Segmentation
------------
SegmentationStrategy       – abstract base
FixedTokenSegmentation     – fixed M-token steps (SPO-compatible)
StopSequenceSegmentation   – step boundaries at explicit stop strings
AdaptiveTokenSegmentation  – decaying token budget per depth
"""

from .branching import (
    BranchingStrategy,
    FixedBranchingStrategy,
    EntropyBranchingStrategy,
    BRANCH_MIN,
    BRANCH_MAX,
)
from .termination import (
    TerminationStrategy,
    DepthTerminationStrategy,
    PDegradationTerminationStrategy,
    CombinedTerminationStrategy,
)
from .segmentation import (
    SegmentationStrategy,
    FixedTokenSegmentation,
    StopSequenceSegmentation,
    AdaptiveTokenSegmentation,
)

__all__ = [
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
