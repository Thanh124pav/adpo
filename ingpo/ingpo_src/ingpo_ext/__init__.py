"""InGPO extension package.

Importing the package only loads the dependency-free `core` modules.  The
SPO-coupled inference strategy and episode generator must be activated
explicitly via :func:`register_with_treetune` so that unit tests for the
core modules can run in environments where the full SPO source tree is not
on PYTHONPATH.
"""

from ingpo_ext.core import (
    log_prob_matrix,
    segment_index,
    tv_distance,
    thresholds,
    answer_set,
    lp_scorer,
    triggers,
    vllm_scorer,
)


def register_with_treetune() -> None:
    """Import the SPO-dependent modules so their @register decorators run."""

    # Import for side effects only.
    from ingpo_ext.inference_strategies import ingpo_inference_strategy  # noqa: F401
    from ingpo_ext.episode_generators import ingpo_episode_generator  # noqa: F401
