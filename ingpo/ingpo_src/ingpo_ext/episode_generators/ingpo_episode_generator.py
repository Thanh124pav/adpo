"""InGPO episode generator.

Subclasses `HybridEpisodeGenerator` from SPO and only overrides
`extract_edges_from_tree` so we can:

  * skip pruned children (no edge — segment contributes nothing to PPO),
  * for shared children, inherit `value` / `reward` from the share target
    instead of using the segment's own (NaN) reward,
  * carry through `ingpo_action`, `ingpo_share_target`, `ingpo_tv_m`,
    `ingpo_gap_m` so downstream metric loggers can quote them.

Everything else — replay buffer, _add_logprobs_to_edges, PPO collation —
is inherited unchanged so the trainer behaviour is identical to SPO.

The corresponding `*_strategy` registration lets configs use
`type: 'ingpo_episode_generator'`.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from treetune.episode_generators import EpisodeGenerator
from treetune.episode_generators.hybrid_episode_generator import (
    HybridEpisodeGenerator,
)
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


@EpisodeGenerator.register("ingpo_episode_generator")
class InGPOEpisodeGenerator(HybridEpisodeGenerator):
    """Tree → edges with online Share/Prune awareness."""

    def __init__(
        self,
        ingpo_zero_advantage_when_pruned: bool = True,
        ingpo_emit_pruned_edges: bool = False,
        ingpo_share_inherit: str = "value_and_reward",  # or "value_only"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ingpo_zero_advantage_when_pruned = ingpo_zero_advantage_when_pruned
        self.ingpo_emit_pruned_edges = ingpo_emit_pruned_edges
        self.ingpo_share_inherit = ingpo_share_inherit

    # ------------------------------------------------------------------
    # Override edge extraction
    # ------------------------------------------------------------------

    def extract_edges_from_tree(
        self,
        tree,
        adv_method: str = "rloo",
        only_adv_greater_than_zero: bool = True,
        use_hard_estimation: bool = False,
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        data_instance = tree["_request_object"]
        question_id = data_instance["_treetune__idx"]

        # Index every node so SHARE children can dereference their target.
        index_by_seg_id: Dict[str, Dict[str, Any]] = {}

        def collect(node):
            seg_id = node.get("ingpo_segment_id")
            if seg_id is not None:
                index_by_seg_id[seg_id] = node
            for ch in node.get("children", []) or []:
                collect(ch)

        tree_copy = copy.deepcopy(tree)
        collect(tree_copy)

        ingpo_stats = tree_copy.get("ingpo_stats", {})
        if ingpo_stats:
            self._cloud_log({**ingpo_stats, "ingpo/answer_set_size": tree_copy.get("ingpo_answer_set_size", 0)})

        def BoK(value, bok=4):
            return 1 - (1 - value) ** bok

        def resolve_reward(node, fallback_parent_reward: float) -> Optional[float]:
            r = node.get("reward")
            if r is None:
                return None
            if isinstance(r, float) and np.isnan(r):
                action = node.get("ingpo_action")
                if action == "share":
                    target_id = node.get("ingpo_share_target")
                    if target_id is not None and target_id in index_by_seg_id:
                        target_r = index_by_seg_id[target_id].get("reward")
                        if target_r is not None and not (isinstance(target_r, float) and np.isnan(target_r)):
                            return float(target_r)
                    return float(fallback_parent_reward)
                if action == "prune":
                    return float(fallback_parent_reward)
                return None
            return float(r)

        def dfs(node, parent=None):
            if parent is not None:
                query_text = parent["full_text"]
                response_text = node["text"]
                parent_reward = parent.get("reward", 0.0) or 0.0
                parent_reward_std = parent.get("reward_std", 0.0) or 0.0

                child_reward = resolve_reward(node, parent_reward)
                if child_reward is None:
                    # Unable to resolve; skip edge to avoid corrupting PPO.
                    pass
                else:
                    leaf = node.get("leaf", False)
                    ingpo_action = node.get("ingpo_action", "expand")
                    is_pruned = ingpo_action == "prune"
                    is_shared = ingpo_action == "share"

                    if is_pruned and not self.ingpo_emit_pruned_edges:
                        # Drop the edge entirely - PPO does not see it.
                        pass
                    else:
                        if adv_method == "rloo":
                            advantage = child_reward - parent_reward
                        elif adv_method == "grpo":
                            advantage = (child_reward - parent_reward) / (parent_reward_std + 1e-8)
                        else:
                            raise ValueError(f"adv_method {adv_method} is not supported")

                        if is_pruned and self.ingpo_zero_advantage_when_pruned:
                            advantage = 0.0

                        prover_advantage = BoK(child_reward) - BoK(parent_reward)
                        pav_advantage = advantage + prover_advantage

                        keep = True
                        if only_adv_greater_than_zero and pav_advantage == 0 and not is_pruned:
                            keep = False

                        if keep and len(response_text) > 0:
                            edges.append(
                                {
                                    "question_id": question_id,
                                    "instance": data_instance,
                                    "query_text": query_text,
                                    "response_text": response_text,
                                    "advantage": advantage,
                                    "prover_advantage": prover_advantage,
                                    "value": child_reward,
                                    "leaf": leaf,
                                    "reward": child_reward,
                                    "ingpo_action": ingpo_action,
                                    "ingpo_share_target": node.get("ingpo_share_target"),
                                    "ingpo_tv_m": node.get("ingpo_tv_m"),
                                    "ingpo_gap_m": node.get("ingpo_gap_m"),
                                }
                            )

            for child in node.get("children", []) or []:
                dfs(child, node)
            node.pop("children", None)

        dfs(tree_copy)
        # Round-trip through json so downstream Datasets.from_list works
        # (matches SPO behaviour).
        edges = json.loads(json.dumps(edges, default=lambda o: None))
        return edges
