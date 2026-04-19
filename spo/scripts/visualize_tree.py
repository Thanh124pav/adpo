"""
Visualize SPO tree structure and advantage estimation from saved episodes.

Usage:
    # Visualize a specific iteration's episodes
    python spo/scripts/visualize_tree.py \
        --episodes SPO-experiments/rho1.1b-spo-tree-666/iteration__0/episodes \
        --idx 0

    # Show all samples in an iteration
    python spo/scripts/visualize_tree.py \
        --episodes SPO-experiments/rho1.1b-spo-tree-666/iteration__0/episodes \
        --all

Requires: datasets, rich  (pip install datasets rich)
"""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


# ── helpers ──────────────────────────────────────────────────────────────────

def _color_reward(r):
    if r is None:
        return "[dim]?[/dim]"
    if r >= 0.9:
        return f"[bold green]{r:.3f}[/bold green]"
    if r >= 0.5:
        return f"[yellow]{r:.3f}[/yellow]"
    return f"[red]{r:.3f}[/red]"


def _color_advantage(a):
    if a is None:
        return "[dim]?[/dim]"
    if a > 0.05:
        return f"[bold green]{a:+.3f}[/bold green]"
    if a < -0.05:
        return f"[bold red]{a:+.3f}[/bold red]"
    return f"[dim]{a:+.3f}[/dim]"


def _truncate(text, max_len=120):
    text = text.replace("\n", "↵ ")
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


# ── tree printer ─────────────────────────────────────────────────────────────

def print_tree(node, console, prefix="", is_last=True, parent_reward=None):
    connector = "└── " if is_last else "├── "
    depth = node.get("depth", 0)

    reward = node.get("reward")
    adv = (reward - parent_reward) if (reward is not None and parent_reward is not None) else None
    leaf = node.get("leaf", False)
    finish = node.get("finish_reason", "")

    node_text = node.get("text", "")
    # For root, text is full prompt — skip printing it
    if depth == 0:
        console.print(f"[bold cyan]ROOT[/bold cyan]  reward={_color_reward(reward)}")
    else:
        tag = "[bold magenta][LEAF][/bold magenta]" if leaf else "[dim][INT][/dim] "
        fr = f"  [dim]finish={finish}[/dim]" if finish else ""
        console.print(
            f"{prefix}{connector}{tag} "
            f"reward={_color_reward(reward)}  adv={_color_advantage(adv)}"
            f"{fr}"
        )
        console.print(f"{prefix}{'    ' if is_last else '│   '}[dim]{_truncate(node_text)}[/dim]")

    children = node.get("children", [])
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        print_tree(
            child,
            console,
            prefix=child_prefix,
            is_last=(i == len(children) - 1),
            parent_reward=reward,
        )


# ── chain printer ─────────────────────────────────────────────────────────────

def print_chain(node, console):
    """For SPO-chain: flat root with children = MC rollouts from cutpoints."""
    console.print(f"[bold cyan]ROOT[/bold cyan]  reward={_color_reward(node.get('reward'))}")
    children = node.get("children", [])
    for i, child in enumerate(children):
        r = child.get("reward")
        pr = node.get("reward")
        adv = (r - pr) if (r is not None and pr is not None) else None
        console.print(
            f"  [{i}] reward={_color_reward(r)}  adv={_color_advantage(adv)}  "
            f"[dim]{_truncate(child.get('text', ''))}[/dim]"
        )


# ── summary stats ────────────────────────────────────────────────────────────

def summarize(node, stats=None):
    if stats is None:
        stats = {"rewards": [], "advantages": [], "leaves": 0, "internal": 0}
    r = node.get("reward")
    leaf = node.get("leaf", False)
    if r is not None:
        stats["rewards"].append(r)
    if leaf:
        stats["leaves"] += 1
    else:
        stats["internal"] += 1
    parent_r = node.get("reward")
    for child in node.get("children", []):
        cr = child.get("reward")
        if cr is not None and parent_r is not None:
            stats["advantages"].append(cr - parent_r)
        summarize(child, stats)
    return stats


# ── main ─────────────────────────────────────────────────────────────────────

def show_sample(ds, idx, console):
    row = ds[idx]
    tree_col = "_treetune__reasoning_tree"

    console.rule(f"[bold]Sample #{idx}[/bold]")

    # Print question
    query = row.get("query", row.get("problem", row.get("question", "")))
    console.print(f"[bold]Question:[/bold] {_truncate(query, 300)}\n")

    if tree_col not in row or not row[tree_col]:
        console.print("[red]No tree found in this sample.[/red]")
        return

    tree = json.loads(row[tree_col])
    max_depth = tree.get("depth", 0)

    # Detect tree vs chain by checking if children have children
    children = tree.get("children", [])
    has_grandchildren = any(c.get("children") for c in children)
    mode = "tree" if has_grandchildren else "chain"

    console.print(f"[dim]mode={mode}  children={len(children)}[/dim]\n")

    if mode == "tree":
        print_tree(tree, console)
    else:
        print_chain(tree, console)

    # Stats
    stats = summarize(tree)
    if stats["rewards"]:
        import statistics
        console.print(
            f"\n[bold]Stats:[/bold]  "
            f"leaves={stats['leaves']}  internal={stats['internal']}  "
            f"reward_mean={statistics.mean(stats['rewards']):.3f}  "
            f"reward_std={statistics.stdev(stats['rewards']) if len(stats['rewards']) > 1 else 0:.3f}"
        )
    if stats["advantages"]:
        import statistics
        console.print(
            f"           "
            f"adv_mean={statistics.mean(stats['advantages']):+.3f}  "
            f"adv_std={statistics.stdev(stats['advantages']) if len(stats['advantages']) > 1 else 0:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="Path to saved HF Dataset (save_to_disk)")
    parser.add_argument("--idx", type=int, default=0, help="Sample index to show")
    parser.add_argument("--all", action="store_true", help="Show all samples")
    parser.add_argument("--max", type=int, default=10, help="Max samples when --all")
    args = parser.parse_args()

    try:
        from rich.console import Console
    except ImportError:
        print("Install rich: pip install rich")
        return

    console = Console()

    path = Path(args.episodes)
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        return

    ds = load_from_disk(str(path))
    console.print(f"Loaded {len(ds)} episodes from [cyan]{path}[/cyan]\n")
    console.print(f"Columns: {ds.column_names}\n")

    if args.all:
        for i in range(min(len(ds), args.max)):
            show_sample(ds, i, console)
            console.print()
    else:
        show_sample(ds, args.idx, console)


if __name__ == "__main__":
    main()
