"""
TPO Episode Generator — SPO-compatible with one critical modification:

    **Early-stopped leaf nodes (terminated by P-degradation) get V = 0.**

Inherits directly from SPO's ``TreeEpisodeGeneratorForMath`` and overrides
only the score computation and path extraction, keeping the full PPO training
pipeline (tokenisation, per-token advantage mapping, episode creation)
identical to SPO.

Architecture
------------
``TreeEpisodeGeneratorForMath`` (SPO)
    └── ``TPOEpisodeGenerator``  (this file)
            compute_score  ← sets V=0 for early-stopped nodes
            is_answer_correct ← uses grade_answer via task
            (everything else unchanged)

The generator is registered under ``"tpo"`` with treetune's registry
when the treetune library is importable.

V computation rules
-------------------
- Natural leaf (finish_reason == EOS, ``early_stopped=False``) :
      V = grade_answer(predicted, gold)  ∈ {0, 1}
- Early-stopped leaf (``early_stopped=True``) :
      V = 0.0   ← hard zero, regardless of whether an answer was extracted
- Internal node :
      V = mean(V of children)
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mc_analysis._answer_utils import extract_answer as _default_extract_answer

logger = logging.getLogger(__name__)

TREE_COLNAME = "_treetune__reasoning_tree"

# ---------------------------------------------------------------------------
# Optional treetune registration
# ---------------------------------------------------------------------------

try:
    from treetune.episode_generators import EpisodeGenerator
    from treetune.episode_generators.tree_episode_generator import (
        TreeEpisodeGeneratorForMath,
        TreeEpisodeUtils,
    )
    from treetune.episode_generators.base_episode_generator import Episode
    from treetune.episode_generators.path_filters import SuccessfulPathFilter
    from treetune.tasks import Task
    from treetune.tokenization_utils import Tokenizer
    from datasets import Dataset

    _HAS_TREETUNE = True

    @EpisodeGenerator.register("tpo", exist_ok=True)
    class TPOEpisodeGenerator(TreeEpisodeGeneratorForMath):
        """TPO episode generator: same as SPO's tree generator but V=0 for early-stopped leaves.

        All constructor parameters and training logic are inherited from
        ``TreeEpisodeGeneratorForMath``.  Only ``compute_score`` is overridden.

        Parameters
        ----------
        (all parameters identical to TreeEpisodeGeneratorForMath)
        """

        # ------------------------------------------------------------------
        # Score computation (overrides SPO's TreeEpisodeUtils.compute_score)
        # ------------------------------------------------------------------

        def is_answer_correct(self, task: "Task", node: Dict[str, Any], gold_answer: str) -> float:
            """Grade a leaf node's extracted answer.

            Returns 1.0 if correct, 0.0 otherwise (handles None safely).
            """
            pred = self.extract_pred_answer(task, node)
            if pred is None:
                return 0.0
            return float(task.grade_answer(given_answer=pred, ground_truth=gold_answer))

        def compute_score(
            self,
            task: "Task",
            root: Dict[str, Any],
            gold_answer: str,
        ) -> Dict[str, Any]:
            """Bottom-up V computation with TPO's early-stopping rule.

            Leaf V:
                * ``early_stopped=True``  → 0.0
                * natural EOS             → grade_answer(predicted, gold)

            Internal V:
                * mean(V of children)
            """
            def _dfs(node: Dict[str, Any]) -> float:
                if "answer" in node:
                    if node.get("early_stopped", False):
                        # P-degradation terminated → V = 0 (TPO spec)
                        node["score"] = 0.0
                    else:
                        node["score"] = self.is_answer_correct(task, node, gold_answer)
                    node["is_correct_answer"] = node["score"]
                    return node["score"]

                child_scores = [_dfs(c) for c in node.get("children", [])]
                node["score"] = float(np.mean(child_scores)) if child_scores else 0.0
                return node["score"]

            _dfs(root)
            return root

except ImportError:
    _HAS_TREETUNE = False
    logger.warning(
        "[TPOEpisodeGenerator] treetune not available — "
        "TPOEpisodeGenerator will not be registered in the treetune registry. "
        "Use the standalone helper functions below instead."
    )


# ---------------------------------------------------------------------------
# Standalone helpers (no treetune required)
# Used when running TPO outside the full SPO/treetune training stack.
# ---------------------------------------------------------------------------

def compute_score_tpo(
    root: Dict[str, Any],
    gold_answer: str,
    answer_checker: Callable[[Optional[str], str], bool],
    extract_answer_fn: Callable[[str], Optional[str]] = _default_extract_answer,
) -> Dict[str, Any]:
    """Bottom-up V computation (TPO rule) — standalone, no treetune needed.

    Parameters
    ----------
    root : dict
        TPO tree root as returned by :func:`tpo.algorithm.build_tree_tpo`.
    gold_answer : str
        Reference answer.
    answer_checker : callable
        ``checker(predicted_str, gold_str) → bool``.
    extract_answer_fn : callable, optional
        ``text → answer_str | None``.

    Returns
    -------
    dict
        *root* in-place annotated with ``node["score"]`` for every node.
    """
    def _dfs(node: Dict[str, Any]) -> float:
        if "answer" in node:
            if node.get("early_stopped", False):
                # TPO rule: P-degradation leaf → V = 0
                node["score"] = 0.0
            else:
                pred = extract_answer_fn(node.get("answer", ""))
                node["score"] = float(
                    pred is not None and answer_checker(pred, gold_answer)
                )
            node["is_correct_answer"] = node["score"]
            return node["score"]

        child_scores = [_dfs(c) for c in node.get("children", [])]
        node["score"] = float(np.mean(child_scores)) if child_scores else 0.0
        return node["score"]

    _dfs(root)
    return root


def compute_advantage_tpo(root: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-node advantage = V(child) - V(parent).

    Identical to SPO's ``TreeEpisodeUtils.compute_advantage``.
    Stored in ``node["advantage"]`` for every non-root node.
    """
    def _dfs(node: Dict[str, Any]) -> None:
        if "answer" in node:
            return
        for child in node.get("children", []):
            child["advantage"] = child["score"] - node["score"]
            _dfs(child)

    _dfs(root)
    return root


def extract_paths_tpo(
    root: Dict[str, Any],
    repeat_early_stopped: bool = False,
    branch_factor: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Extract root-to-leaf paths from a TPO tree.

    Each path is a list of nodes from root to leaf, preserving the full
    context chain.  Mirrors ``TreeEpisodeUtils.extract_paths_from_tree``
    but returns node lists directly (no dict wrapper).

    Parameters
    ----------
    repeat_early_stopped : bool
        If True, repeat early-stopped paths ``branch_factor^(max_depth-depth)``
        times (SPO's importance-weighting trick).  Requires *branch_factor*
        and *max_depth*.
    branch_factor : int, optional
        Used for importance weighting when *repeat_early_stopped* is True.
    max_depth : int, optional
        Used for importance weighting when *repeat_early_stopped* is True.

    Returns
    -------
    List[List[Node]]
        Each inner list is a root-to-leaf path (inclusive).
    """
    paths: List[List[Dict[str, Any]]] = []

    def _dfs(node: Dict[str, Any], depth: int, chain: List[Dict[str, Any]]) -> None:
        current_chain = chain + [node]
        if "answer" in node:
            if repeat_early_stopped and max_depth is not None and branch_factor is not None:
                weight = branch_factor ** (max_depth - depth)
                paths.extend([current_chain] * max(1, weight))
            else:
                paths.append(current_chain)
            return
        for child in node.get("children", []):
            _dfs(child, depth + 1, current_chain)

    _dfs(root, 0, [])
    return paths


def tree_to_episodes_standalone(
    tree_json: str,
    gold_answer: str,
    tokenizer,
    answer_checker: Callable[[Optional[str], str], bool],
    extract_answer_fn: Callable[[str], Optional[str]] = _default_extract_answer,
    append_eos: bool = True,
) -> List[Dict[str, Any]]:
    """Convert a JSON-serialised TPO tree into training episodes.

    Standalone alternative to ``TPOEpisodeGenerator.convert_to_episode``
    for use outside the treetune framework (e.g. with verl / TRL).

    Each episode dict contains:
        ``query_token_ids``    : List[int]
        ``response_token_ids`` : List[int]
        ``advantages``         : List[float]  (per-token, same as SPO)
        ``score``              : float         (leaf correctness)
        ``early_stopped``      : bool
        ``query_text``         : str
        ``response_text``      : str

    Per-token advantages are computed by distributing each node's advantage
    uniformly over all tokens in that node's text span, identical to SPO's
    ``convert_path_to_episode`` logic.

    Parameters
    ----------
    tree_json : str
        JSON string from the ``_treetune__reasoning_tree`` column.
    gold_answer : str
        Reference answer (for scoring leaf correctness).
    tokenizer
        HuggingFace tokenizer with ``encode``, ``decode``, ``eos_token_id``.
    answer_checker : callable
        ``(predicted, gold) → bool``.
    extract_answer_fn : callable
        ``text → str | None``.
    append_eos : bool
        Whether to append EOS token to natural (non-early-stopped) leaves.
    """
    root = json.loads(tree_json)

    # 1. Score + advantage
    compute_score_tpo(root, gold_answer, answer_checker, extract_answer_fn)
    compute_advantage_tpo(root)

    # 2. Extract paths
    paths = extract_paths_tpo(root)
    if not paths:
        return []

    episodes = []
    for path in paths:
        # path[0] = root (prompt only, no text contribution)
        # path[1:] = reasoning steps
        query_text = path[0]["text"]
        leaf = path[-1]
        full_text = leaf["full_text"]
        response_text = full_text[len(query_text):]
        is_early_stopped = leaf.get("early_stopped", False)
        score = leaf["score"]

        # Tokenise
        full_enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        query_enc = tokenizer(query_text, return_offsets_mapping=True, add_special_tokens=False)
        response_enc = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)

        query_ids: List[int] = full_enc["input_ids"][:len(query_enc["input_ids"])]
        response_ids: List[int] = full_enc["input_ids"][len(query_enc["input_ids"]):]

        if not response_ids:
            continue

        # Per-character advantage array for the response
        adv_char = np.zeros(len(response_text), dtype=np.float32)
        offset = 0
        for node in path[1:]:  # skip root
            node_text = node["text"]
            adv = node.get("advantage", 0.0)
            length = len(node_text)
            adv_char[offset: offset + length] = adv
            offset += length

        # Map character advantages to token advantages
        offsets = response_enc["offset_mapping"]
        adv_token = np.array(
            [adv_char[min(off[0], len(adv_char) - 1)] for off in offsets],
            dtype=np.float32,
        )

        if append_eos and not is_early_stopped and tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]
            adv_token = np.append(adv_token, adv_token[-1] if len(adv_token) > 0 else 0.0)

        episodes.append({
            "query_text": query_text,
            "response_text": response_text,
            "query_token_ids": query_ids,
            "response_token_ids": response_ids,
            "advantages": adv_token.tolist(),
            "score": float(score),
            "early_stopped": bool(is_early_stopped),
        })

    return episodes
