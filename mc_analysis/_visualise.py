"""
Tree visualisation helpers.

    print_tree(root)              – ASCII/Unicode summary to stdout
    visualise(root, ...)          – matplotlib figure
"""
import textwrap
from typing import Any, Dict, Optional

Node = Dict[str, Any]


# ── text output ───────────────────────────────────────────────────────────────

def _jsd_str(node: Node) -> str:
    """Compact JSD display: 'JSD={n2:0.30, n3:0.00}'."""
    jsd: Dict[str, float] = node.get("JSD", {})
    if not jsd:
        return "JSD={}"
    inner = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(jsd.items()))
    return f"JSD={{{inner}}}"


def print_tree(root: Node, indent: int = 0, width: int = 80) -> None:
    """
    Print a compact ASCII representation of the annotated tree.

    Each node shows:
        [name]  V=<value>  P=<value>  JSD={sibling: score, …}
        <first line of node text (truncated)>
    """
    prefix = "  " * indent
    name   = node.get("name", "?")
    v_str  = f"V={node['V']:.3f}"   if "V" in node  else "V=?"
    p_str  = f"P={node['P']:.2f}"   if "P" in node  else "P=?"
    jsd_s  = _jsd_str(node)
    correct_mark = ""
    if "correct" in node:
        correct_mark = " ✓" if node["correct"] else " ✗"

    header = f"{prefix}[{name}]  {v_str}  {p_str}  {jsd_s}{correct_mark}"
    print(header)

    text = node.get("text", "")
    snippet = textwrap.shorten(text.replace("\n", " "), width=width - len(prefix) - 2)
    print(f"{prefix}  {snippet!r}")

    for child in node.get("children", []):
        print_tree(child, indent=indent + 1, width=width)


# ── matplotlib visualisation ──────────────────────────────────────────────────

def _collect_nodes(root: Node):
    """Return all nodes in BFS order."""
    from collections import deque
    order = []
    q = deque([root])
    while q:
        n = q.popleft()
        order.append(n)
        q.extend(n.get("children", []))
    return order


def _assign_layout(root: Node):
    """
    Assign (x, y) coordinates using a simple Reingold-Tilford-like algorithm.

    Leaves are placed left-to-right with unit spacing.
    Inner nodes are centred above their children.
    y = -depth  (so root is at top).
    """
    leaves = []

    def _collect_leaves(n: Node):
        if not n.get("children"):
            leaves.append(n)
        for c in n.get("children", []):
            _collect_leaves(c)

    _collect_leaves(root)
    for i, lf in enumerate(leaves):
        lf["_x"] = float(i)

    def _set_inner_x(n: Node):
        children = n.get("children", [])
        if not children:
            return
        for c in children:
            _set_inner_x(c)
        n["_x"] = (children[0]["_x"] + children[-1]["_x"]) / 2.0

    def _set_y(n: Node, depth: int = 0):
        n["_y"] = float(-depth)
        for c in n.get("children", []):
            _set_y(c, depth + 1)

    _set_inner_x(root)
    _set_y(root)


def visualise(
    root: Node,
    figsize=(14, 8),
    node_width: float = 2.2,
    node_height: float = 1.1,
    fontsize: int = 7,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Draw the annotated tree as a matplotlib figure.

    Node colour encodes V (green = correct, red = wrong) via RdYlGn.
    Each box shows: name, V, P, JSD (Dict), and a text snippet.

    Parameters
    ----------
    root : Node
        Annotated tree root (output of any ``analyse*`` function).
    figsize : tuple
        Figure size passed to ``plt.figure``.
    node_width, node_height : float
        Box dimensions in data coordinates.
    fontsize : int
        Font size for node labels.
    title : str, optional
        Figure title.
    show : bool
        Call ``plt.show()`` at the end.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    _assign_layout(root)
    all_nodes = _collect_nodes(root)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    cmap      = plt.get_cmap("RdYlGn")
    norm      = Normalize(vmin=0.0, vmax=1.0)
    sm        = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    hw = node_width  / 2.0
    hh = node_height / 2.0

    # Draw edges first (so boxes sit on top)
    for node in all_nodes:
        px, py = node.get("_x", 0), node.get("_y", 0)
        for child in node.get("children", []):
            cx, cy = child.get("_x", 0), child.get("_y", 0)
            ax.plot([px, cx], [py - hh, cy + hh], color="#888888", lw=0.8, zorder=0)

    # Draw nodes
    for node in all_nodes:
        x, y  = node.get("_x", 0), node.get("_y", 0)
        v_val = node.get("V", 0.5)
        color = cmap(norm(v_val))

        rect = mpatches.FancyBboxPatch(
            (x - hw, y - hh), node_width, node_height,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#444444", linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(rect)

        # Build label
        name   = node.get("name", "?")
        v_str  = f"V={v_val:.3f}"
        p_str  = f"P={node['P']:.2f}" if "P" in node else ""
        jsd    = node.get("JSD", {})
        if jsd:
            jsd_lines = "  ".join(f"{k}:{v:.2f}" for k, v in sorted(jsd.items()))
            jsd_str   = f"JSD: {jsd_lines}"
        else:
            jsd_str = ""

        text_snippet = textwrap.shorten(
            node.get("text", "").replace("\n", " "),
            width=30,
        )

        lines = [f"[{name}]  {v_str}", p_str, jsd_str, f'"{text_snippet}"']
        label = "\n".join(ln for ln in lines if ln)

        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=fontsize,
            wrap=True,
            zorder=2,
        )

    # Auto-scale axes
    xs = [n.get("_x", 0) for n in all_nodes]
    ys = [n.get("_y", 0) for n in all_nodes]
    pad = max(node_width, node_height)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if title:
        fig.suptitle(title, fontsize=10)

    plt.colorbar(sm, ax=ax, shrink=0.4, label="V (correctness)")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig
