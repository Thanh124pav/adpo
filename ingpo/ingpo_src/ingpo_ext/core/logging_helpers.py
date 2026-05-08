"""Pure helpers used by the InGPO episode generator's wandb logging.

Lives in `core/` (no SPO deps) so it can be unit-tested without installing
the full SPO Python stack.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


_DEMO_TEXT_TRUNC = 240


DEMO_COLUMNS = [
    "question_id", "action", "depth", "seg_id",
    "parent_text", "child_text", "target_text", "target_seg_id",
    "avg_lp_K", "tv_m", "gap_m", "eta", "tau",
]


def truncate(s: Optional[str], n: int = _DEMO_TEXT_TRUNC) -> str:
    if not s:
        return ""
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 3] + "..."


def per_depth_action_counts(tree) -> Dict[str, float]:
    """Walk the tree and return per-depth share/prune/expand counts and
    rates as a flat dict of wandb-friendly metric names.
    """

    per_depth_count: Dict[int, Counter] = {}
    stack = [tree]
    while stack:
        n = stack.pop()
        d = n.get("ingpo_depth")
        a = n.get("ingpo_action")
        if d is not None and a is not None:
            per_depth_count.setdefault(d, Counter())[a] += 1
        stack.extend(n.get("children") or [])

    out: Dict[str, float] = {}
    for d, c in sorted(per_depth_count.items()):
        total = sum(c.values())
        if total == 0:
            continue
        out[f"ingpo/depth_{d}/n"] = total
        out[f"ingpo/depth_{d}/expand_count"] = c.get("expand", 0)
        out[f"ingpo/depth_{d}/share_count"] = c.get("share", 0)
        out[f"ingpo/depth_{d}/prune_count"] = c.get("prune", 0)
        out[f"ingpo/depth_{d}/share_rate"] = c.get("share", 0) / total
        out[f"ingpo/depth_{d}/prune_rate"] = c.get("prune", 0) / total
    return out


def collect_demo_rows(
    tree,
    index_by_seg_id: Dict[str, Dict[str, Any]],
    question_id,
    n_each: int,
) -> Dict[str, List[List[Any]]]:
    """Return up to `n_each` SHARE demos and `n_each` PRUNE demos, each as a
    list of column-aligned cells matching `DEMO_COLUMNS`.
    """

    prune_rows: List[List[Any]] = []
    share_rows: List[List[Any]] = []

    stack = [tree]
    while stack:
        n = stack.pop()
        stack.extend(n.get("children") or [])
        action = n.get("ingpo_action")
        if action not in ("share", "prune"):
            continue

        seg_id = n.get("ingpo_segment_id", "")
        depth = n.get("ingpo_depth", "")
        parent_id = n.get("ingpo_parent_segment_id", "")
        parent = index_by_seg_id.get(parent_id, {})
        target_id = n.get("ingpo_share_target")
        target = index_by_seg_id.get(target_id, {}) if target_id else {}

        row = [
            str(question_id),
            action,
            int(depth) if isinstance(depth, int) else depth,
            str(seg_id),
            truncate(parent.get("text") or parent.get("full_text")),
            truncate(n.get("text") or n.get("full_text")),
            truncate(target.get("text") or target.get("full_text")) if target_id else "",
            str(target_id or ""),
            float(n.get("ingpo_avg_lp_K") or 0.0),
            float(n.get("ingpo_tv_m") or 0.0) if n.get("ingpo_tv_m") is not None else None,
            float(n.get("ingpo_gap_m") or 0.0) if n.get("ingpo_gap_m") is not None else None,
            float(n.get("ingpo_eta") or 0.0),
            float(n.get("ingpo_tau") or 0.0),
        ]
        (prune_rows if action == "prune" else share_rows).append(row)

    return {
        "share": share_rows[:n_each],
        "prune": prune_rows[:n_each],
    }


def row_to_dict(row: List[Any]) -> Dict[str, Any]:
    return dict(zip(DEMO_COLUMNS, row))


def render_md_section(
    tree_idx: int,
    question_id,
    stats: Dict[str, Any],
    demo_rows: Dict[str, List[List[Any]]],
) -> str:
    """One Markdown section for one tree, ready to append to demos.md."""

    out = [f"## Tree #{tree_idx}  (question_id={question_id})\n"]
    if stats:
        share_rate = float(stats.get("ingpo/share_rate", 0.0) or 0.0)
        prune_rate = float(stats.get("ingpo/prune_rate", 0.0) or 0.0)
        out.append(
            f"- share_rate: **{share_rate:.3f}**, prune_rate: **{prune_rate:.3f}**, "
            f"#shared={stats.get('ingpo/shared_count', 0)}, "
            f"#pruned={stats.get('ingpo/pruned_count', 0)}, "
            f"#expanded={stats.get('ingpo/expanded_count', 0)}\n"
        )

    for label, rows in (("SHARE", demo_rows["share"]), ("PRUNE", demo_rows["prune"])):
        if not rows:
            continue
        out.append(f"### {label} demos\n")
        for row in rows:
            d = row_to_dict(row)
            line = (
                f"- depth={d['depth']}  seg={d['seg_id']}  "
                f"AvgLP_K={float(d['avg_lp_K']):.3f}  "
            )
            if d.get("tv_m") is not None:
                line += f"TV_m={float(d['tv_m']):.3f}  "
            if d.get("gap_m") is not None:
                line += f"gap_m={float(d['gap_m']):.3f}  "
            line += f"(eta={float(d['eta']):.3f}, tau={float(d['tau']):.3f})\n"
            out.append(line)
            out.append(f"  - parent : `{d['parent_text']}`\n")
            out.append(f"  - child  : `{d['child_text']}`\n")
            if d.get("target_text"):
                out.append(
                    f"  - shared->`{d['target_seg_id']}` : `{d['target_text']}`\n"
                )
        out.append("\n")
    out.append("\n")
    return "".join(out)


def to_jsonl_record(
    tree_idx: int,
    question_id,
    answer_set_size: int,
    stats: Dict[str, Any],
    per_depth: Dict[str, float],
    demo_rows: Dict[str, List[List[Any]]],
) -> Dict[str, Any]:
    """Pack one tree's metrics + demos into a JSONL-ready dict."""

    return {
        "tree_idx": tree_idx,
        "question_id": question_id,
        "answer_set_size": answer_set_size,
        "stats": stats,
        "per_depth": per_depth,
        "demos": {
            "share": [row_to_dict(r) for r in demo_rows["share"]],
            "prune": [row_to_dict(r) for r in demo_rows["prune"]],
        },
    }
