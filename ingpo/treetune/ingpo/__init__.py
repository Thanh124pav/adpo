"""InGPO algorithm components — formerly the `ingpo_ext` package.

Loading this package only pulls in dependency-free core modules.  The
`InGPOEpisodeGenerator` / `InGPOInferenceStrategy` are registered via
``treetune.episode_generators`` / ``treetune.inference_strategies`` so that
configs with ``type: 'ingpo'`` resolve automatically.
"""

from treetune.ingpo import (
    answer_set,
    local_value_share,
    log_prob_matrix,
    logging_helpers,
    lp_scorer,
    segment_index,
    thresholds,
    triggers,
    tv_distance,
    vllm_scorer,
)
