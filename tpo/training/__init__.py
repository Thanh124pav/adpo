"""
TPO training module.

Importing this package registers the TPO classes with treetune's registry
(when treetune is importable), making them available in JSONNET configs:

    inference_strategy: { type: "tpo", ... }
    episode_generator:  { type: "tpo", ... }
"""

from .inference_strategy import TPOInferenceStrategy, TREE_COLNAME
from .episode_generator import (
    compute_score_tpo,
    compute_advantage_tpo,
    extract_paths_tpo,
    tree_to_episodes_standalone,
)

# Import TPOEpisodeGenerator if treetune is available (triggers registration)
try:
    from .episode_generator import TPOEpisodeGenerator
except ImportError:
    TPOEpisodeGenerator = None  # type: ignore

__all__ = [
    "TPOInferenceStrategy",
    "TPOEpisodeGenerator",
    "TREE_COLNAME",
    "compute_score_tpo",
    "compute_advantage_tpo",
    "extract_paths_tpo",
    "tree_to_episodes_standalone",
]
