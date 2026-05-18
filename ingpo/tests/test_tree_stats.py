"""Tests for aggregate and per-depth InGPO tree stats."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LOGGING_HELPERS = (
    Path(__file__).resolve().parents[1]
    / "ingpo_src"
    / "ingpo_ext"
    / "core"
    / "logging_helpers.py"
)
_SPEC = importlib.util.spec_from_file_location("ingpo_logging_helpers", _LOGGING_HELPERS)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

aggregate_tree_stats = _MODULE.aggregate_tree_stats
per_depth_action_counts = _MODULE.per_depth_action_counts


def _make_mixed_tree():
    return {
        "ingpo_action": "expand",
        "ingpo_depth": 0,
        "children": [
            {
                "ingpo_action": "expand",
                "ingpo_depth": 1,
                "children": [
                    {"ingpo_action": "expand", "ingpo_depth": 2},
                    {"ingpo_action": "prune", "ingpo_depth": 2},
                ],
            },
            {"ingpo_action": "share", "ingpo_depth": 1},
        ],
    }


def test_aggregate_tree_stats_counts_all_actions():
    stats = aggregate_tree_stats(_make_mixed_tree())
    assert stats["ingpo/expanded_count"] == 3
    assert stats["ingpo/shared_count"] == 1
    assert stats["ingpo/pruned_count"] == 1
    assert stats["ingpo/share_rate"] == 1 / 5
    assert stats["ingpo/prune_rate"] == 1 / 5


def test_aggregate_consistent_with_per_depth():
    tree = _make_mixed_tree()
    agg = aggregate_tree_stats(tree)
    per_depth = per_depth_action_counts(tree)

    sum_expand = sum(v for k, v in per_depth.items() if k.endswith("/expand_count"))
    sum_share = sum(v for k, v in per_depth.items() if k.endswith("/share_count"))
    sum_prune = sum(v for k, v in per_depth.items() if k.endswith("/prune_count"))

    assert agg["ingpo/expanded_count"] == sum_expand
    assert agg["ingpo/shared_count"] == sum_share
    assert agg["ingpo/pruned_count"] == sum_prune


def test_local_value_share_path_keeps_expanded_in_denominator():
    tree = {
        "ingpo_action": "expand",
        "children": [
            {"ingpo_action": "expand"},
            {"ingpo_action": "expand"},
            {"ingpo_action": "share"},
        ],
    }
    stats = aggregate_tree_stats(tree)
    assert stats["ingpo/expanded_count"] == 3
    assert stats["ingpo/shared_count"] == 1
    assert stats["ingpo/share_rate"] == 0.25
    assert stats["ingpo/prune_rate"] == 0.0


def test_empty_tree_returns_empty_dict():
    assert aggregate_tree_stats({}) == {}
    assert aggregate_tree_stats({"children": [{}]}) == {}
