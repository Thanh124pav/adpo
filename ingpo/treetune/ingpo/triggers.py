"""Online Share / Prune triggers for one InGPO tree.

Holds a `LogProbMatrix`, a `SegmentBST`, and the threshold config for a single
problem instance, and exposes the two decision points called inside the tree
expansion loop:

    decide(segment_id, prefix, parent_id) -> Decision

The decision flow follows PLAN Algorithm 3:

  1. Score K fast indices for the new segment s.
  2. FindNearest in the BST. If close enough, fill full m and check TV_m.
     If TV_m <= eta -> SHARE with that segment.
  3. Else compare AvgLP_K(s) with AvgLP_K(pa). If much lower, fill full and
     check AvgLP_m gap. If still lower by eta -> PRUNE.
  4. Else INSERT into BST and let the caller expand children.

Per the ablation matrix in PLAN.md we expose `enable_share` and
`enable_prune` flags so a single class can drive Share-only / Prune-only /
Both runs.

`share_target` controls Abl 3: "parent" / "root" / "nearest".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from treetune.ingpo.answer_set import AnswerSet
from treetune.ingpo.log_prob_matrix import LogProbMatrix, SegmentLP
from treetune.ingpo.lp_scorer import LPScorer
from treetune.ingpo.segment_index import SegmentBST
from treetune.ingpo.thresholds import ThresholdConfig, compute_eta, compute_tau
from treetune.ingpo.tv_distance import (
    avg_lp_diff_K,
    conditional_ig_lower_bound,
    tv_m,
)


class Action(str, Enum):
    EXPAND = "expand"
    SHARE = "share"
    PRUNE = "prune"


@dataclass
class Decision:
    action: Action
    segment_id: str
    parent_id: Optional[str]
    share_target: Optional[str] = None
    avg_lp_K: float = 0.0
    avg_lp_m: Optional[float] = None
    tv_m: Optional[float] = None
    avg_lp_diff_to_pa_K: Optional[float] = None
    avg_lp_diff_to_pa_m: Optional[float] = None
    eta_used: float = 0.0
    tau_used: float = 0.0


@dataclass
class TriggerStats:
    expanded: int = 0
    shared: int = 0
    pruned: int = 0
    avg_tv_share: float = 0.0
    avg_gap_prune: float = 0.0

    def update_share(self, tv: float) -> None:
        n = self.shared + 1
        self.avg_tv_share = (self.avg_tv_share * self.shared + tv) / n
        self.shared = n

    def update_prune(self, gap: float) -> None:
        n = self.pruned + 1
        self.avg_gap_prune = (self.avg_gap_prune * self.pruned + gap) / n
        self.pruned = n

    def as_dict(self) -> Dict[str, float]:
        total = self.expanded + self.shared + self.pruned
        share_rate = self.shared / max(total, 1)
        prune_rate = self.pruned / max(total, 1)
        return {
            "ingpo/expanded_count": self.expanded,
            "ingpo/shared_count": self.shared,
            "ingpo/pruned_count": self.pruned,
            "ingpo/share_rate": share_rate,
            "ingpo/prune_rate": prune_rate,
            "ingpo/avg_tv_when_share": self.avg_tv_share,
            "ingpo/avg_gap_when_prune": self.avg_gap_prune,
        }


@dataclass
class TriggerEngine:
    answer_set: AnswerSet
    scorer: LPScorer
    thresholds: ThresholdConfig
    enable_share: bool = True
    enable_prune: bool = True
    share_target: str = "nearest"  # one of {"nearest", "parent", "root"}
    root_segment_id: str = "root"
    lp_matrix: LogProbMatrix = field(init=False)
    bst: SegmentBST = field(init=False)
    stats: TriggerStats = field(default_factory=TriggerStats)

    def __post_init__(self) -> None:
        K = self.thresholds.K
        m = max(self.answer_set.m, 1)
        if K > m:
            K = m
            self.thresholds.K = K
        self.lp_matrix = LogProbMatrix(K=K, m=m)
        self.bst = SegmentBST()

    # ---- public API -------------------------------------------------------

    async def register_root(self, prefix: str) -> SegmentLP:
        """Score K fast indices for the root and seed the BST."""

        return await self._score_and_register(
            segment_id=self.root_segment_id,
            prefix=prefix,
        )

    async def decide(
        self,
        *,
        segment_id: str,
        parent_id: str,
        prefix: str,
        is_leaf: bool,
    ) -> Decision:
        """Score the new segment and pick Action.SHARE / PRUNE / EXPAND."""

        row_s = await self._score_and_register(segment_id=segment_id, prefix=prefix)
        eta = compute_eta(self.thresholds, delta_avg=self.lp_matrix.avg_delta())
        tau = compute_tau(self.thresholds, eta)

        decision = Decision(
            action=Action.EXPAND,
            segment_id=segment_id,
            parent_id=parent_id,
            avg_lp_K=row_s.avg_lp_K,
            eta_used=eta,
            tau_used=tau,
        )

        # Leaves don't need triggers - they are terminals; just bookkeep.
        if is_leaf:
            self.stats.expanded += 1
            self.bst.insert(row_s.avg_lp_K, segment_id)
            return decision

        # ---- Trigger 1: ValueShare ---------------------------------------
        if self.enable_share:
            target_id = self._pick_share_target(parent_id, row_s.avg_lp_K)
            if target_id is not None and target_id != segment_id:
                row_t = self.lp_matrix.get(target_id)
                gap_K = avg_lp_diff_K(row_s, row_t)
                if gap_K < tau:
                    await self._fill_full([row_s, row_t], prefixes=[prefix, None])
                    tv = tv_m(row_s, row_t)
                    if tv <= eta:
                        decision.action = Action.SHARE
                        decision.share_target = target_id
                        decision.tv_m = tv
                        decision.avg_lp_m = row_s.avg_lp_m
                        # we still insert into BST so the cluster grows
                        self.bst.insert(row_s.avg_lp_K, segment_id)
                        self.stats.update_share(tv)
                        return decision

        # ---- Trigger 2: Prune --------------------------------------------
        if self.enable_prune and parent_id is not None and self.lp_matrix.has(parent_id):
            row_pa = self.lp_matrix.get(parent_id)
            gap_K = row_pa.avg_lp_K - row_s.avg_lp_K
            decision.avg_lp_diff_to_pa_K = gap_K
            if gap_K > tau:
                await self._fill_full([row_s, row_pa], prefixes=[prefix, None])
                gap_m = row_pa.avg_lp_m - row_s.avg_lp_m
                decision.avg_lp_diff_to_pa_m = gap_m
                decision.avg_lp_m = row_s.avg_lp_m
                if gap_m > eta:
                    decision.action = Action.PRUNE
                    self.stats.update_prune(gap_m)
                    return decision

        # ---- Default: expand ---------------------------------------------
        self.bst.insert(row_s.avg_lp_K, segment_id)
        self.stats.expanded += 1
        return decision

    # ---- internals --------------------------------------------------------

    async def _score_and_register(self, segment_id: str, prefix: str) -> SegmentLP:
        if self.lp_matrix.has(segment_id):
            return self.lp_matrix.get(segment_id)
        K = self.lp_matrix.K
        fast = await self.scorer.score_batch(
            segment_id=segment_id,
            prefix=prefix,
            y=self.answer_set.y,
            indices=list(range(K)),
        )
        return self.lp_matrix.add_row(segment_id, fast, prefix=prefix)

    async def _fill_full(self, rows: Sequence[SegmentLP], prefixes: Sequence[Optional[str]]):
        K = self.lp_matrix.K
        m = self.lp_matrix.m
        if K >= m:
            for r in rows:
                if r.full is None:
                    r.full = r.fast.copy()
            return

        for row, prefix in zip(rows, prefixes):
            if row.full is not None:
                continue
            use_prefix = prefix or row.prefix
            if use_prefix is None:
                raise RuntimeError(
                    f"Need prefix to fill full LP for segment {row.segment_id}"
                )
            tail = await self.scorer.score_batch(
                segment_id=row.segment_id,
                prefix=use_prefix,
                y=self.answer_set.y,
                indices=list(range(K, m)),
            )
            self.lp_matrix.fill_full(row.segment_id, tail)

    def _pick_share_target(
        self, parent_id: Optional[str], avg_lp_K: float
    ) -> Optional[str]:
        if self.share_target == "parent":
            return parent_id
        if self.share_target == "root":
            return self.root_segment_id
        # default: nearest in BST
        nearest = self.bst.find_nearest(avg_lp_K)
        if nearest is None:
            return None
        return nearest[1]
