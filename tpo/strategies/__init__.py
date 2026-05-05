"""
TPO strategy plug-ins.

Branching
---------
BranchingStrategy         – abstract base
FixedBranchingStrategy    – constant branch factor (SPO-compatible)
EntropyBranchingStrategy  – dynamic branching from next-token entropy [1, 8]

Termination
-----------
TerminationStrategy              – abstract base
DepthTerminationStrategy         – max-depth cap (SPO-compatible)
PDegradationTerminationStrategy  – terminates when P(answer|traj) degrades K times
CombinedTerminationStrategy      – OR of multiple strategies
MLClassifierTermination          – learned classifier over exp(log_P) values

Segmentation
------------
SegmentationStrategy        – abstract base
FixedTokenSegmentation      – fixed M-token steps (SPO-compatible)
StopSequenceSegmentation    – step boundaries at explicit stop strings
AdaptiveTokenSegmentation   – decaying token budget per depth
DelayBranchingSegmentation  – generate long chunk, truncate at high-uncertainty position
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
    MLClassifierTermination,
)
from .segmentation import (
    SegmentationStrategy,
    FixedTokenSegmentation,
    StopSequenceSegmentation,
    AdaptiveTokenSegmentation,
    DelayBranchingSegmentation,
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
    "MLClassifierTermination",
    # segmentation
    "SegmentationStrategy",
    "FixedTokenSegmentation",
    "StopSequenceSegmentation",
    "AdaptiveTokenSegmentation",
    "DelayBranchingSegmentation",
]
