"""Tests for the InGPO wandb logging helpers (per-depth aggregates +
prune/share demo rows). These run on the dependency-free `core/` module
so they don't need SPO/transformers/etc."""

from __future__ import annotations

from treetune.ingpo.logging_helpers import (
    DEMO_COLUMNS,
    collect_demo_rows,
    per_depth_action_counts,
    truncate,
)


def make_tree():
    # depth-0 root, two depth-1 children (one expands, one shares root),
    # under the expanded one: one prune + one expand at depth 2.
    root = {
        "ingpo_segment_id": "root",
        "ingpo_action": "expand",
        "ingpo_depth": 0,
        "text": "ROOT TEXT",
        "full_text": "ROOT TEXT",
        "children": [],
    }
    a = {
        "ingpo_segment_id": "root/0/0",
        "ingpo_action": "expand",
        "ingpo_depth": 1,
        "ingpo_parent_segment_id": "root",
        "text": "expanded child A",
        "full_text": "ROOT TEXT expanded child A",
        "ingpo_avg_lp_K": -1.2,
        "ingpo_eta": 0.02,
        "ingpo_tau": 0.05,
        "children": [],
    }
    b = {
        "ingpo_segment_id": "root/0/1",
        "ingpo_action": "share",
        "ingpo_depth": 1,
        "ingpo_parent_segment_id": "root",
        "ingpo_share_target": "root",
        "text": "duplicate-of-root child B",
        "full_text": "ROOT TEXT duplicate-of-root child B",
        "ingpo_avg_lp_K": -1.1,
        "ingpo_tv_m": 0.01,
        "ingpo_eta": 0.02,
        "ingpo_tau": 0.05,
    }
    a_child0 = {
        "ingpo_segment_id": "root/0/0/1/0",
        "ingpo_action": "prune",
        "ingpo_depth": 2,
        "ingpo_parent_segment_id": "root/0/0",
        "text": "way-off-track",
        "full_text": "... way-off-track",
        "ingpo_avg_lp_K": -7.0,
        "ingpo_gap_m": 5.5,
        "ingpo_eta": 0.02,
        "ingpo_tau": 0.05,
    }
    a_child1 = {
        "ingpo_segment_id": "root/0/0/1/1",
        "ingpo_action": "expand",
        "ingpo_depth": 2,
        "ingpo_parent_segment_id": "root/0/0",
        "text": "ok-second-step",
        "full_text": "... ok-second-step",
    }
    a["children"] = [a_child0, a_child1]
    root["children"] = [a, b]
    index = {
        n["ingpo_segment_id"]: n
        for n in [root, a, b, a_child0, a_child1]
    }
    return root, index


def test_per_depth_action_counts():
    tree, _ = make_tree()
    out = per_depth_action_counts(tree)

    # depth 0: just root, expand=1
    assert out["ingpo/depth_0/n"] == 1
    assert out["ingpo/depth_0/expand_count"] == 1
    assert out["ingpo/depth_0/share_count"] == 0
    assert out["ingpo/depth_0/prune_count"] == 0

    # depth 1: 1 expand + 1 share
    assert out["ingpo/depth_1/n"] == 2
    assert out["ingpo/depth_1/share_count"] == 1
    assert out["ingpo/depth_1/expand_count"] == 1
    assert out["ingpo/depth_1/share_rate"] == 0.5

    # depth 2: 1 expand + 1 prune
    assert out["ingpo/depth_2/n"] == 2
    assert out["ingpo/depth_2/prune_count"] == 1
    assert out["ingpo/depth_2/prune_rate"] == 0.5


def test_collect_demo_rows_picks_share_and_prune():
    tree, index = make_tree()
    rows = collect_demo_rows(tree, index, question_id="q-7", n_each=4)

    assert len(rows["share"]) == 1
    assert len(rows["prune"]) == 1

    # Schema width matches column header.
    for r in rows["share"] + rows["prune"]:
        assert len(r) == len(DEMO_COLUMNS)

    share = rows["share"][0]
    qid_idx = DEMO_COLUMNS.index("question_id")
    action_idx = DEMO_COLUMNS.index("action")
    target_idx = DEMO_COLUMNS.index("target_seg_id")
    parent_idx = DEMO_COLUMNS.index("parent_text")
    child_idx = DEMO_COLUMNS.index("child_text")

    assert share[qid_idx] == "q-7"
    assert share[action_idx] == "share"
    assert share[target_idx] == "root"
    assert "ROOT TEXT" in share[parent_idx]
    assert "duplicate-of-root child B" in share[child_idx]

    prune = rows["prune"][0]
    assert prune[action_idx] == "prune"
    # PRUNE child has no share_target.
    assert prune[target_idx] == ""
    assert "way-off-track" in prune[child_idx]


def test_collect_demo_rows_respects_n_each_cap():
    # Three SHARE children under root.
    tree = {
        "ingpo_segment_id": "root",
        "ingpo_action": "expand",
        "ingpo_depth": 0,
        "children": [
            {
                "ingpo_segment_id": f"r/0/{i}",
                "ingpo_action": "share",
                "ingpo_depth": 1,
                "ingpo_parent_segment_id": "root",
                "ingpo_share_target": "root",
                "text": f"share #{i}",
            }
            for i in range(3)
        ],
    }
    index = {tree["ingpo_segment_id"]: tree}
    rows = collect_demo_rows(tree, index, question_id="qid", n_each=1)
    assert len(rows["share"]) == 1
    assert len(rows["prune"]) == 0


def test_truncate():
    assert truncate(None) == ""
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdefghij", 6) == "abc..."
    assert truncate("a\nb") == "a \\n b"


def test_render_md_section_share_and_prune():
    from treetune.ingpo.logging_helpers import render_md_section, to_jsonl_record

    tree, index = make_tree()
    rows = collect_demo_rows(tree, index, question_id="q-7", n_each=4)
    md = render_md_section(
        tree_idx=3,
        question_id="q-7",
        stats={"ingpo/share_rate": 0.5, "ingpo/prune_rate": 0.5,
               "ingpo/shared_count": 1, "ingpo/pruned_count": 1,
               "ingpo/expanded_count": 2},
        demo_rows=rows,
    )
    assert "Tree #3" in md
    assert "share_rate: **0.500**" in md
    assert "prune_rate: **0.500**" in md
    assert "### SHARE demos" in md
    assert "### PRUNE demos" in md
    # Per-row content is included.
    assert "duplicate-of-root child B" in md
    assert "way-off-track" in md
    assert "shared->`root`" in md


def test_render_md_section_no_demos_still_safe():
    from treetune.ingpo.logging_helpers import render_md_section

    md = render_md_section(
        tree_idx=0,
        question_id="qid",
        stats={},
        demo_rows={"share": [], "prune": []},
    )
    assert md.startswith("## Tree #0")
    # No demo subsections when there are no rows.
    assert "### SHARE" not in md
    assert "### PRUNE" not in md


def test_to_jsonl_record_roundtrip():
    import json as _json
    from treetune.ingpo.logging_helpers import to_jsonl_record

    tree, index = make_tree()
    rows = collect_demo_rows(tree, index, question_id="q-7", n_each=2)
    record = to_jsonl_record(
        tree_idx=11,
        question_id="q-7",
        answer_set_size=42,
        stats={"ingpo/share_rate": 0.5},
        per_depth={"ingpo/depth_1/share_rate": 0.5},
        demo_rows=rows,
    )
    # Must be JSON-serialisable.
    blob = _json.dumps(record, default=str)
    parsed = _json.loads(blob)
    assert parsed["tree_idx"] == 11
    assert parsed["answer_set_size"] == 42
    # demo dict has share/prune lists with column-keyed entries.
    assert isinstance(parsed["demos"]["share"], list)
    if parsed["demos"]["share"]:
        s = parsed["demos"]["share"][0]
        assert s["action"] == "share"
        assert "child_text" in s
        assert "parent_text" in s
    if parsed["demos"]["prune"]:
        p = parsed["demos"]["prune"][0]
        assert p["action"] == "prune"
