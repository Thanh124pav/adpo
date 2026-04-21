"""
adpo.vllm_helpers — compatibility shim.

All symbols now live in adpo.mc_analysis.vllm_helpers.
Import from there directly, or use ``adpo.mc_analysis`` for the full pipeline.
"""
from adpo.mc_analysis.vllm_helpers import (  # noqa: F401
    rollout_with_alpha,
    compute_sequence_logprob,
    build_tree,
    build_tree_async,
    extract_leaves,
    extract_all_paths,
    cumulative_logprob,
)
