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


from ingpo_ext.core.logging_helpers import (
    DEMO_COLUMNS,
    collect_demo_rows,
    per_depth_action_counts,
    render_md_section,
    to_jsonl_record,
)


@EpisodeGenerator.register("ingpo_episode_generator")
class InGPOEpisodeGenerator(HybridEpisodeGenerator):
    """Tree → edges with online Share/Prune awareness."""

    def __init__(
        self,
        ingpo_zero_advantage_when_pruned: bool = True,
        ingpo_emit_pruned_edges: bool = False,
        ingpo_share_inherit: str = "value_and_reward",  # or "value_only"
        ingpo_demo_examples_per_tree: int = 2,  # how many SHARE / PRUNE demos to log per tree
        ingpo_demos_dir: Optional[str] = None,  # absolute path; else exp_root/ingpo_demos
        ingpo_log_demos_to_wandb: bool = False,  # for offline servers, default off
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ingpo_zero_advantage_when_pruned = ingpo_zero_advantage_when_pruned
        self.ingpo_emit_pruned_edges = ingpo_emit_pruned_edges
        self.ingpo_share_inherit = ingpo_share_inherit
        self.ingpo_demo_examples_per_tree = int(ingpo_demo_examples_per_tree)
        self.ingpo_log_demos_to_wandb = bool(ingpo_log_demos_to_wandb)
        # Where to dump local-file demos. Resolves on first use because
        # exp_root is only set after super().__init__ on some SPO branches.
        self._ingpo_demos_dir_override = ingpo_demos_dir
        self._ingpo_demos_dir_resolved = None  # type: Optional[Any]
        self._ingpo_jsonl_handle = None
        self._ingpo_md_handle = None
        self._tree_seen = 0

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
        per_depth = self._per_depth_action_counts(tree_copy)
        demo_rows = collect_demo_rows(
            tree_copy,
            index_by_seg_id,
            question_id=question_id,
            n_each=max(self.ingpo_demo_examples_per_tree, 0),
        )
        self._tree_seen += 1

        # ---- Local-file demo dump (works offline, no wandb required) -----
        self._dump_demos_to_disk(
            tree_idx=self._tree_seen,
            question_id=question_id,
            stats=ingpo_stats,
            per_depth=per_depth,
            demo_rows=demo_rows,
            answer_set_size=tree_copy.get("ingpo_answer_set_size", 0),
        )

        # ---- Optional wandb scalar+table logging --------------------------
        if ingpo_stats or per_depth:
            log_entry = {
                **ingpo_stats,
                "ingpo/answer_set_size": tree_copy.get("ingpo_answer_set_size", 0),
                "ingpo/tree_idx": self._tree_seen,
                **per_depth,
                "ingpo/n_share_demos_in_tree": len(demo_rows["share"]),
                "ingpo/n_prune_demos_in_tree": len(demo_rows["prune"]),
            }
            if self.ingpo_log_demos_to_wandb:
                table = self._maybe_build_wandb_table(demo_rows)
                if table is not None:
                    log_entry["ingpo/demos"] = table
            self._cloud_log(log_entry)

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

    # ------------------------------------------------------------------
    # Logging helpers (thin wrappers around module-level pure helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _per_depth_action_counts(tree) -> Dict[str, float]:
        return per_depth_action_counts(tree)

    def _maybe_build_wandb_table(self, demo_rows):
        if not (demo_rows["share"] or demo_rows["prune"]):
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            return None
        table = wandb.Table(columns=DEMO_COLUMNS)
        for r in demo_rows["share"] + demo_rows["prune"]:
            table.add_data(*r)
        return table

    # ------------------------------------------------------------------
    # Offline-friendly file dump
    # ------------------------------------------------------------------

    def _resolve_demos_dir(self):
        if self._ingpo_demos_dir_resolved is not None:
            return self._ingpo_demos_dir_resolved

        from pathlib import Path

        if self._ingpo_demos_dir_override:
            base = Path(self._ingpo_demos_dir_override)
        elif getattr(self, "exp_root", None) is not None:
            base = Path(self.exp_root) / "ingpo_demos"
        else:
            base = Path.cwd() / "ingpo_demos"

        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"InGPO: could not create demos dir {base}: {exc}")
            self._ingpo_demos_dir_resolved = False
            return False
        self._ingpo_demos_dir_resolved = base
        return base

    def _open_demo_handles(self):
        base = self._resolve_demos_dir()
        if base is False:
            return None, None
        if self._ingpo_jsonl_handle is None:
            self._ingpo_jsonl_handle = (base / "demos.jsonl").open("a", buffering=1)
        if self._ingpo_md_handle is None:
            self._ingpo_md_handle = (base / "demos.md").open("a", buffering=1)
            if self._ingpo_md_handle.tell() == 0:
                self._ingpo_md_handle.write(
                    "# InGPO SHARE / PRUNE demos\n\n"
                    "One section per tree. `tail -F demos.md` to follow live.\n\n"
                )
        return self._ingpo_jsonl_handle, self._ingpo_md_handle

    def _dump_demos_to_disk(
        self,
        tree_idx: int,
        question_id,
        stats: Dict[str, Any],
        per_depth: Dict[str, float],
        demo_rows: Dict[str, List[List[Any]]],
        answer_set_size: int,
    ) -> None:
        if max(self.ingpo_demo_examples_per_tree, 0) == 0 and not stats:
            return

        jsonl, md = self._open_demo_handles()
        if jsonl is None:
            return  # silent: never block training because logging failed

        # JSONL row: machine-readable. One line per tree.
        record = to_jsonl_record(
            tree_idx=tree_idx,
            question_id=question_id,
            answer_set_size=answer_set_size,
            stats=stats,
            per_depth=per_depth,
            demo_rows=demo_rows,
        )
        try:
            jsonl.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"InGPO: failed to append demos.jsonl: {exc}")

        # Markdown: human-readable. One section per tree, only if there's
        # something interesting (rates or demos) to show.
        if not (demo_rows["share"] or demo_rows["prune"]):
            return
        try:
            md.write(render_md_section(tree_idx, question_id, stats, demo_rows))
        except Exception as exc:
            logger.warning(f"InGPO: failed to append demos.md: {exc}")

    def __del__(self):
        for h in (self._ingpo_jsonl_handle, self._ingpo_md_handle):
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass
