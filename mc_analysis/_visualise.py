"""
Tree visualisation helpers.

    print_tree(root)              – ASCII/Unicode summary to stdout
    visualise(root, ...)          – matplotlib figure
    visualise_html(root, path)    – standalone interactive HTML (D3.js)
"""
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

Node = Dict[str, Any]


# ── text output ───────────────────────────────────────────────────────────────

def _jsd_str(node: Node) -> str:
    """Compact JSD display: 'JSD={n2:0.30, n3:0.00}'."""
    jsd: Dict[str, float] = node.get("JSD", {})
    if not jsd:
        return "JSD={}"
    inner = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(jsd.items()))
    return f"JSD={{{inner}}}"


def print_tree(node: Node, indent: int = 0, width: int = 80) -> None:
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


# ── HTML / D3 visualisation ───────────────────────────────────────────────────

def _node_to_d3(node: Node) -> Dict[str, Any]:
    """Convert an annotated Node to a plain dict suitable for D3 hierarchy."""
    name  = node.get("name", "?")
    v     = node.get("V")
    p     = node.get("P")
    jsd   = node.get("JSD", {})
    text  = textwrap.shorten(node.get("text", "").replace("\n", " "), width=120)

    label_parts = [f"<b>{name}</b>"]
    if v is not None:
        label_parts.append(f"V={v:.3f}")
    if p is not None:
        label_parts.append(f"P={p:.2f}")

    jsd_lines = "  ".join(f"{k}:{s:.3f}" for k, s in sorted(jsd.items()))
    if jsd_lines:
        label_parts.append(f"JSD: {jsd_lines}")

    d: Dict[str, Any] = {
        "name":    name,
        "label":   " | ".join(label_parts),
        "tooltip": text,
        "V":       round(v, 4) if v is not None else None,
        "P":       round(p, 4) if p is not None else None,
        "correct": node.get("correct"),
    }
    children = node.get("children", [])
    if children:
        d["children"] = [_node_to_d3(c) for c in children]
    return d


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #1a1a2e; font-family: monospace; color: #eee; }}
  h2   {{ text-align: center; padding: 10px; margin: 0;
          font-size: 14px; color: #a0c4ff; }}
  #tree-container {{ width: 100vw; overflow-x: auto; }}
  svg  {{ display: block; }}
  .node circle {{
    stroke-width: 1.5px;
    cursor: pointer;
  }}
  .node text {{
    font-size: 11px;
    fill: #ddd;
  }}
  .link {{
    fill: none;
    stroke: #555;
    stroke-width: 1px;
  }}
  .tooltip {{
    position: absolute;
    background: rgba(0,0,0,.85);
    border: 1px solid #444;
    padding: 6px 10px;
    font-size: 11px;
    pointer-events: none;
    max-width: 400px;
    word-wrap: break-word;
    border-radius: 4px;
    color: #eee;
    display: none;
  }}
  .legend {{ text-align: center; font-size: 12px; padding: 4px; color: #aaa; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div class="legend">Node colour: <span style="color:#d73027">V=0 (wrong)</span>
 → <span style="color:#fee08b">V=0.5</span>
 → <span style="color:#1a9850">V=1 (correct)</span></div>
<div id="tree-container"></div>
<div class="tooltip" id="tt"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = {data_json};

// ── colour scale: 0→red, 0.5→yellow, 1→green ────────────────────────────────
const colour = d3.scaleSequential()
  .domain([0, 1])
  .interpolator(d3.interpolateRdYlGn);

// ── layout ───────────────────────────────────────────────────────────────────
const nodeW = 200, nodeH = 70, hGap = 40, vGap = 90;

const root = d3.hierarchy(DATA);
const treeLayout = d3.tree()
  .nodeSize([nodeW + hGap, vGap]);

treeLayout(root);

// Shift so leftmost x >= 0
const xs = root.descendants().map(d => d.x);
const xMin = Math.min(...xs);
root.descendants().forEach(d => {{ d.x -= xMin; }});

const xMax = Math.max(...root.descendants().map(d => d.x));
const yMax = Math.max(...root.descendants().map(d => d.y));
const svgW = xMax + nodeW + 60;
const svgH = yMax + nodeH + 60;

const svg = d3.select("#tree-container").append("svg")
  .attr("width",  svgW)
  .attr("height", svgH);

const g = svg.append("g").attr("transform", "translate(30,30)");

// ── links ────────────────────────────────────────────────────────────────────
g.selectAll(".link")
  .data(root.links())
  .join("path")
  .attr("class", "link")
  .attr("d", d3.linkVertical()
    .x(d => d.x + nodeW / 2)
    .y(d => d.y + nodeH / 2));

// ── nodes ────────────────────────────────────────────────────────────────────
const node = g.selectAll(".node")
  .data(root.descendants())
  .join("g")
  .attr("class", "node")
  .attr("transform", d => `translate(${{d.x}},${{d.y}})`);

// Box
node.append("rect")
  .attr("width",  nodeW)
  .attr("height", nodeH)
  .attr("rx", 6)
  .attr("fill",   d => d.data.V != null ? colour(d.data.V) : "#555")
  .attr("stroke", d => {{
    if (d.data.correct === true)  return "#00ff88";
    if (d.data.correct === false) return "#ff4444";
    return "#888";
  }})
  .attr("stroke-width", d => d.data.correct != null ? 2.5 : 1);

// Label (HTML via foreignObject for multi-line)
node.append("foreignObject")
  .attr("width",  nodeW)
  .attr("height", nodeH)
  .append("xhtml:div")
  .style("font-size",   "10px")
  .style("padding",     "4px 6px")
  .style("line-height", "1.4")
  .style("color",       d => (d.data.V != null && d.data.V > 0.45 && d.data.V < 0.85) ? "#222" : "#111")
  .style("overflow",    "hidden")
  .html(d => d.data.label);

// ── tooltip ──────────────────────────────────────────────────────────────────
const tt = d3.select("#tt");
node
  .on("mouseover", (evt, d) => {{
    tt.style("display", "block")
      .html(`<b>${{d.data.name}}</b><br>${{d.data.tooltip}}`);
  }})
  .on("mousemove", evt => {{
    tt.style("left", (evt.pageX + 12) + "px")
      .style("top",  (evt.pageY - 20) + "px");
  }})
  .on("mouseout", () => tt.style("display", "none"));
</script>
</body>
</html>
"""


def visualise_html(
    root: Node,
    path: str,
    title: Optional[str] = None,
) -> str:
    """
    Save the annotated tree as a self-contained interactive HTML file.

    Uses D3.js (loaded from CDN) to render a vertical tree layout.
    Nodes are colour-coded by V (red→yellow→green), leaf borders are
    green (correct) or red (wrong).  Hover a node to see its full text.

    Parameters
    ----------
    root : Node
        Annotated tree root (output of any ``analyse*`` function).
    path : str
        Output file path, e.g. ``"./results/tree.html"``.
    title : str, optional
        Page / chart title.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    from pathlib import Path as _Path

    title   = title or f"mc_analysis — {root.get('name', 'root')}  V={root.get('V', '?'):.3f}"
    d3_data = json.dumps(_node_to_d3(root), ensure_ascii=False)
    html    = _HTML_TEMPLATE.format(title=title, data_json=d3_data)

    out = _Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)
