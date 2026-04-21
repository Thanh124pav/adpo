"""
adpo.mc_analysis
================

Monte-Carlo tree analysis for mathematical reasoning chains.

Quick start (vLLM backend)
--------------------------
    from adpo.mc_analysis import analyse, print_tree, visualise

    root = analyse(
        question    = "What is 2 + 2?",
        gold_answer = "4",
        server_url  = "http://localhost:8000/v1",
        model_name  = "Qwen/Qwen2.5-7B-Instruct",
        tree_kwargs = {"max_depth": 2, "branch_factor": 3},
    )
    print_tree(root)
    visualise(root, title="2+2 tree")

Quick start (HuggingFace backend)
----------------------------------
    from adpo.mc_analysis import HFBackend, analyse_hf, print_tree

    backend = HFBackend("Qwen/Qwen2.5-7B-Instruct", device="cuda")
    root = analyse_hf(
        question    = "What is 2 + 2?",
        gold_answer = "4",
        backend     = backend,
        tree_kwargs = {"max_depth": 2, "branch_factor": 3},
    )
    print_tree(root)

Node fields after analysis
--------------------------
    name     – hierarchical name ("root", "n1", "n1.2", …)
    V        – mean fraction of correct leaves reachable from this node  [0, 1]
    P        – log P(gold_answer | trajectory to node)   [negative float]
    JSD      – {sibling_name: JSD(V_self, V_sibling)}  ({} for root)
    correct  – bool (leaves only)
"""

# ── pipeline ──────────────────────────────────────────────────────────────────
from ._pipeline import (
    analyse,
    analyse_async,
    analyse_hf,
    analyse_hf_async,
)

# ── annotation primitives ─────────────────────────────────────────────────────
from ._annotation import (
    assign_names,
    compute_v,
    compute_jsd,
)

# ── answer utilities ──────────────────────────────────────────────────────────
from ._answer_utils import (
    extract_answer,
    default_answer_checker,
)

# ── visualisation ─────────────────────────────────────────────────────────────
from ._visualise import (
    print_tree,
    visualise,
)

# ── backends ──────────────────────────────────────────────────────────────────
from .hf_helpers import (
    HFBackend,
    rollout_with_alpha_hf,
    compute_sequence_logprob_hf,
    build_tree_hf,
    build_tree_hf_async,
)

from .vllm_helpers import (
    rollout_with_alpha,
    compute_sequence_logprob,
    build_tree,
    build_tree_async,
    extract_leaves,
    extract_all_paths,
    cumulative_logprob,
)

__all__ = [
    # pipeline
    "analyse",
    "analyse_async",
    "analyse_hf",
    "analyse_hf_async",
    # annotation
    "assign_names",
    "compute_v",
    "compute_jsd",
    # answer utils
    "extract_answer",
    "default_answer_checker",
    # visualisation
    "print_tree",
    "visualise",
    # HF backend
    "HFBackend",
    "rollout_with_alpha_hf",
    "compute_sequence_logprob_hf",
    "build_tree_hf",
    "build_tree_hf_async",
    # vLLM backend
    "rollout_with_alpha",
    "compute_sequence_logprob",
    "build_tree",
    "build_tree_async",
    "extract_leaves",
    "extract_all_paths",
    "cumulative_logprob",
]
