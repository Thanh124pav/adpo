"""
mc_analysis.py

Monte Carlo tree analysis pipeline for SPO generation trees.

Pipeline
--------
Given a (question, gold_answer) pair:

1. **Build tree** – sample continuations from vLLM using the SPO build_tree
   algorithm (standalone, no guidance/verl dependency).

2. **Compute P** – for every node compute P(answer | trajectory to node):
   - Leaf nodes    : P = 1.0 if the extracted answer matches gold_answer, else 0.0
   - Internal nodes: P = mean(P of children)   [Monte-Carlo tree estimate]

3. **Compute V** – SPO advantage at every node:
   V(node) = P(node) - P(parent)     (root: V = P(root), no parent baseline)

4. **Compute JSD** – for every set of siblings, compute pairwise
   Jensen-Shannon divergences based on their Bernoulli(P) distributions.
   Each node stores a list JSD(node, sibling_i) for all siblings ≠ node.

5. **Visualise** – draw the annotated tree with matplotlib.  Every node box
   shows: name, truncated text, V, P, JSD sequence.  Node colour encodes P
   via a Red→Yellow→Green colourmap.

Public API
----------
    analyse(question, gold_answer, server_url, model_name, ...) → Node
    analyse_async(...)                                           → Node  (async)
    print_tree(root)                                            (debug text dump)
    visualise(root, output_path=None, ...)                      → Figure
"""

import asyncio
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# ── type alias matching SPO convention ───────────────────────────────────────
Node = Dict[str, Any]

_EPS = 1e-12


# =============================================================================
# Answer utilities (re-uses adpo/reward_functions.py when available)
# =============================================================================

def _try_import_reward_fns():
    try:
        from adpo.reward_functions import extract_boxed_answer, is_equiv
        return extract_boxed_answer, is_equiv
    except ImportError:
        return None, None


_ext_boxed, _is_equiv = _try_import_reward_fns()


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from generated text.

    Tries in order:
    1. ``\\boxed{...}`` (LaTeX, handles nested braces)
    2. Last non-empty line as fallback
    """
    if _ext_boxed is not None:
        ans = _ext_boxed(text)
        if ans is not None:
            return ans
    else:
        # Minimal regex fallback for simple \boxed{...}
        m = re.search(r"\\boxed\{([^{}]*)\}", text)
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def default_answer_checker(generated: Optional[str], gold: str) -> bool:
    """Check if *generated* matches *gold*.

    Uses adpo.reward_functions.is_equiv when available (handles LaTeX,
    numeric, fraction, percentage, …).  Falls back to normalised string
    equality → numeric comparison → sympy.
    """
    gen = (generated or "").strip()
    ref = (gold or "").strip()

    if _is_equiv is not None:
        return bool(_is_equiv(gen, ref))

    # Minimal fallback
    if gen.lower() == ref.lower():
        return True
    try:
        return abs(float(gen) - float(ref)) < 1e-5
    except (ValueError, TypeError):
        pass
    try:
        import sympy
        return bool(sympy.simplify(sympy.sympify(gen) - sympy.sympify(ref)) == 0)
    except Exception:
        pass
    return False


# =============================================================================
# Step 1 – tree construction is delegated to vllm_helpers.build_tree
# =============================================================================
# (imported lazily inside analyse / analyse_async to avoid circular imports)


# =============================================================================
# Step 2 – node naming
# =============================================================================

def _assign_names(node: Node, name: str = "root") -> None:
    """Assign a human-readable hierarchical ``name`` to every node (in-place)."""
    node["name"] = name
    for i, child in enumerate(node.get("children", []), start=1):
        child_name = f"n{i}" if name == "root" else f"{name}.{i}"
        _assign_names(child, child_name)


# =============================================================================
# Step 3 – P(correct answer | trajectory to node)  [bottom-up, MC estimate]
# =============================================================================

def _compute_p(
    node: Node,
    gold_answer: str,
    checker: Callable[[Optional[str], str], bool],
) -> float:
    """Recursively compute and store P on every node.  Returns node["P"]."""
    children = node.get("children", [])

    if not children:
        # Leaf node – evaluate answer correctness
        answer = node.get("answer")           # already extracted by build_tree
        is_correct = bool(answer is not None and checker(answer, gold_answer))
        node["correct"] = is_correct
        node["P"] = float(is_correct)
        return node["P"]

    child_ps = [_compute_p(c, gold_answer, checker) for c in children]
    node["P"] = float(np.mean(child_ps))
    return node["P"]


# =============================================================================
# Step 4 – V = SPO advantage = P(node) - P(parent)  [top-down]
# =============================================================================

def _compute_v(node: Node, parent_p: float = 0.0) -> None:
    """Recursively assign V (SPO advantage) to every node (in-place).

    Root node:  V = P(root)  (no parent baseline → parent_p = 0.0)
    Other nodes: V = P(node) - P(parent)
    """
    node["V"] = node["P"] - parent_p
    for child in node.get("children", []):
        _compute_v(child, node["P"])


# =============================================================================
# Step 5 – JSD between siblings
# =============================================================================

def _binary_jsd(p: float, q: float) -> float:
    """Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q).

    JSD(P ‖ Q) = H(M) - ½[H(P) + H(Q)]
    where  M = (P + Q) / 2  and  H is binary entropy.

    Result is in nats, then normalised to [0, 1] by dividing by ln(2).
    Special cases: JSD = 0 when p == q; JSD = 1 when {p,q} = {0,1}.
    """
    p = float(np.clip(p, _EPS, 1 - _EPS))
    q = float(np.clip(q, _EPS, 1 - _EPS))

    if abs(p - q) < _EPS:
        return 0.0

    def h_bin(x: float) -> float:
        x = np.clip(x, _EPS, 1 - _EPS)
        return float(-x * np.log(x) - (1 - x) * np.log(1 - x))

    m = (p + q) / 2.0
    jsd_nats = h_bin(m) - (h_bin(p) + h_bin(q)) / 2.0
    return float(np.clip(jsd_nats / np.log(2), 0.0, 1.0))


def _compute_jsd(node: Node) -> None:
    """Assign ``node["JSD"]`` = list of pairwise binary JSD scores with siblings.

    For a node N that is the i-th child of its parent P (with siblings S_0…S_{k-1}):
        N["JSD"] = [JSD(N, S_j)  for j ≠ i]   in sibling order

    Root has no siblings → root["JSD"] = [].
    Computation is recursive (visits the whole subtree).
    """
    node.setdefault("JSD", [])          # root initialised here; children below

    children = node.get("children", [])
    if not children:
        return

    ps = [c["P"] for c in children]
    n = len(children)
    for i, child in enumerate(children):
        child["JSD"] = [_binary_jsd(ps[i], ps[j]) for j in range(n) if j != i]

    for child in children:
        _compute_jsd(child)


# =============================================================================
# Public: analyse  (synchronous)
# =============================================================================

def analyse(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    answer_extractor: Optional[Callable[[str], Optional[str]]] = None,
) -> Node:
    """Run the full MC analysis pipeline for one (question, gold_answer) pair.

    Parameters
    ----------
    question : str
        Input question / chat prompt.
    gold_answer : str
        Reference answer used to score leaf nodes.
    server_url : str
        vLLM server base URL, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model identifier registered in the vLLM server.
    tree_kwargs : dict, optional
        Keyword arguments forwarded to :func:`adpo.vllm_helpers.build_tree`.
        Common keys: ``max_depth``, ``branch_factor``, ``temperature``,
        ``max_tokens``, ``stop``.
    answer_checker : callable, optional
        ``(generated: str | None, gold: str) → bool``.
        Defaults to :func:`default_answer_checker` (uses adpo.reward_functions).
    answer_extractor : callable, optional
        ``(text: str) → str | None`` applied by build_tree to each leaf.
        Defaults to :func:`extract_answer` (parses ``\\boxed{...}``).

    Returns
    -------
    Node
        Root of the annotated tree.  Every node has:
        ``name``, ``P``, ``V``, ``JSD`` (and ``correct`` for leaf nodes).
    """
    from adpo.vllm_helpers import build_tree  # lazy import

    checker = answer_checker or default_answer_checker
    extractor = answer_extractor or extract_answer
    kw: Dict[str, Any] = dict(tree_kwargs or {})

    # 1. Build generation tree
    root = build_tree(
        server_url=server_url,
        model_name=model_name,
        prompt=question,
        extract_answer_fn=extractor,
        **kw,
    )

    # 2–5. Annotate
    _assign_names(root)
    _compute_p(root, gold_answer, checker)
    _compute_v(root, parent_p=0.0)
    _compute_jsd(root)

    return root


# =============================================================================
# Public: analyse_async  (asynchronous, concurrent expansion)
# =============================================================================

async def analyse_async(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    answer_extractor: Optional[Callable[[str], Optional[str]]] = None,
) -> Node:
    """Async version of :func:`analyse` (uses :func:`build_tree_async`).

    All parameters are identical to :func:`analyse`.
    ``tree_kwargs`` may additionally contain ``max_concurrent`` (int) to limit
    simultaneous HTTP requests.
    """
    from adpo.vllm_helpers import build_tree_async  # lazy import

    checker = answer_checker or default_answer_checker
    extractor = answer_extractor or extract_answer
    kw: Dict[str, Any] = dict(tree_kwargs or {})

    root = await build_tree_async(
        server_url=server_url,
        model_name=model_name,
        prompt=question,
        extract_answer_fn=extractor,
        **kw,
    )

    _assign_names(root)
    _compute_p(root, gold_answer, checker)
    _compute_v(root, parent_p=0.0)
    _compute_jsd(root)

    return root


# =============================================================================
# Text display utility (for debugging)
# =============================================================================

def print_tree(node: Node, indent: int = 0, max_text: int = 60) -> None:
    """Print a compact text representation of the annotated tree.

    Example output::

        [root]          P=0.667  V=+0.667  JSD=[]
          text: "What is 2+2?"
          [n1]          P=1.000  V=+0.333  JSD=[0.918]
            text: "2+2=4, so the answer is \\boxed{4}."  ✓
          [n2] ✗        P=0.000  V=-0.667  JSD=[0.918]
            text: "I think it's \\boxed{5}."
    """
    name = node.get("name", "?")
    p = node.get("P", 0.0)
    v = node.get("V", 0.0)
    jsd = node.get("JSD", [])

    raw_text = node.get("text", "")
    disp_text = raw_text[:max_text].replace("\n", " ")
    if len(raw_text) > max_text:
        disp_text += "…"

    is_leaf = "answer" in node
    correct_mark = ""
    if is_leaf:
        correct_mark = " ✓" if node.get("correct") else " ✗"

    jsd_str = "[" + ", ".join(f"{j:.3f}" for j in jsd) + "]"
    pad = "  " * indent
    print(f'{pad}[{name}]{correct_mark:<3}  P={p:.3f}  V={v:+.3f}  JSD={jsd_str}')
    print(f'{pad}  text: "{disp_text}"')

    for child in node.get("children", []):
        print_tree(child, indent + 1, max_text)


# =============================================================================
# Visualisation
# =============================================================================

# ── Layout constants (data-coordinate units) ─────────────────────────────────
_NW    = 3.6    # node box width
_NH    = 2.4    # node box height
_H_GAP = 0.5    # horizontal gap between sibling subtrees
_V_GAP = 3.4    # vertical distance between depth levels
_FS    = 7      # base font size (pt)
_MAX_T = 38     # max chars per display line inside a node box


# ── Helpers ───────────────────────────────────────────────────────────────────

def _subtree_w(node: Node) -> float:
    """Width (data units) required by this node's subtree."""
    children = node.get("children", [])
    if not children:
        return _NW + _H_GAP
    return sum(_subtree_w(c) for c in children)


def _do_layout(
    node: Node,
    x_left: float,
    y: float,
    pos: Dict[str, Tuple[float, float]],
) -> None:
    """Recursively assign (x_centre, y) positions."""
    children = node.get("children", [])
    sw = _subtree_w(node)
    pos[node["name"]] = (x_left + sw / 2.0, y)
    cur = x_left
    for child in children:
        _do_layout(child, cur, y - _V_GAP, pos)
        cur += _subtree_w(child)


def _p_to_colour(p: float):
    """Map P ∈ [0,1] to an RGBA colour via the RdYlGn colourmap."""
    from matplotlib.cm import RdYlGn
    return RdYlGn(float(np.clip(p, 0.0, 1.0)))


def _fmt_text(raw: str, max_chars: int = _MAX_T, max_lines: int = 2) -> str:
    """Truncate and word-wrap *raw* to at most *max_lines* lines of *max_chars*."""
    raw = " ".join(raw.split())          # collapse whitespace
    if len(raw) <= max_chars:
        return raw
    wrapped = textwrap.wrap(raw, width=max_chars, max_lines=max_lines,
                            placeholder="…")
    return "\n".join(wrapped)


def _draw_node(ax, node: Node, x: float, y: float) -> None:
    """Draw one node box at position (x, y)."""
    import matplotlib.patches as mpatches

    p     = node.get("P", 0.0)
    v     = node.get("V", 0.0)
    jsd   = node.get("JSD", [])
    name  = node.get("name", "?")
    is_leaf = "answer" in node

    text_raw = node.get("text", "")
    text_disp = _fmt_text(text_raw)

    fc = _p_to_colour(p)
    ec = tuple(max(0.0, c - 0.22) for c in fc[:3]) + (1.0,)

    # ── background box ────────────────────────────────────────────────────────
    box = mpatches.FancyBboxPatch(
        (x - _NW / 2, y - _NH / 2), _NW, _NH,
        boxstyle="round,pad=0.07",
        facecolor=fc, edgecolor=ec,
        linewidth=1.3, zorder=2,
    )
    ax.add_patch(box)

    # ── header: node name + correctness mark ──────────────────────────────────
    if is_leaf:
        mark = " ✓" if node.get("correct") else " ✗"
        mark_col = "#006600" if node.get("correct") else "#990000"
    else:
        mark = ""
        mark_col = "black"

    ax.text(x, y + _NH / 2 - 0.22, name,
            ha="center", va="top", fontsize=_FS + 1.5,
            fontweight="bold", color="black", zorder=3)
    if mark:
        ax.text(x + _NW / 2 - 0.15, y + _NH / 2 - 0.22, mark,
                ha="right", va="top", fontsize=_FS + 1,
                fontweight="bold", color=mark_col, zorder=3)

    # ── separator ─────────────────────────────────────────────────────────────
    sep_y = y + _NH / 2 - 0.46
    ax.plot([x - _NW / 2 + 0.08, x + _NW / 2 - 0.08], [sep_y, sep_y],
            color="black", lw=0.5, alpha=0.45, zorder=3)

    # ── text body (italic, truncated) ─────────────────────────────────────────
    ax.text(x, sep_y - 0.08,
            f'"{text_disp}"',
            ha="center", va="top",
            fontsize=_FS - 1, style="italic", color="#2a2a2a", zorder=3)

    # ── V and P ───────────────────────────────────────────────────────────────
    vp_y = y - _NH / 2 + 0.82
    ax.text(x, vp_y,
            f"V: {v:+.3f}    P: {p:.3f}",
            ha="center", va="top",
            fontsize=_FS, family="monospace", color="black", zorder=3)

    # ── JSD ───────────────────────────────────────────────────────────────────
    if jsd:
        jsd_vals = ", ".join(f"{j:.3f}" for j in jsd)
        # Wrap long lists
        jsd_line = f"JSD: [{jsd_vals}]"
        if len(jsd_line) > _MAX_T + 6:
            jsd_line = "JSD: [" + textwrap.fill(jsd_vals, width=_MAX_T) + "]"
        ax.text(x, vp_y - 0.36,
                jsd_line,
                ha="center", va="top",
                fontsize=max(_FS - 2, 5), family="monospace",
                color="#1a1a1a", zorder=3)


def _all_nodes(root: Node) -> List[Node]:
    """Collect every node in DFS order."""
    out = [root]
    for c in root.get("children", []):
        out.extend(_all_nodes(c))
    return out


# =============================================================================
# Public: visualise
# =============================================================================

def visualise(
    root: Node,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    title: str = "SPO Generation Tree  –  MC Analysis",
) -> "plt.Figure":  # type: ignore[name-defined]
    """Draw the annotated SPO tree and return a matplotlib Figure.

    Each node box contains
    ----------------------
    * **name**  – hierarchical identifier (root, n1, n1.2, …)
    * **text**  – truncated generated continuation (italic)
    * **V**     – SPO advantage  =  P(node) − P(parent)
    * **P**     – P(correct answer | trajectory to node)
    * **JSD**   – list of pairwise Jensen-Shannon divergences with siblings
                  (one value per sibling, in sibling order, range [0, 1])

    Node colour encodes **P** via the Red → Yellow → Green colourmap.
    Leaf nodes additionally show a ✓ / ✗ correctness mark.

    Parameters
    ----------
    root : Node
        Annotated tree returned by :func:`analyse` or :func:`analyse_async`.
    output_path : str, optional
        If given, save the figure to this file path
        (format inferred from extension: .png, .pdf, .svg, …).
    figsize : (float, float), optional
        Override the automatically-computed figure size (width, height) in
        inches.
    dpi : int
        Figure resolution.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (user can call ``plt.show()`` or further customise).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    nodes = _all_nodes(root)

    # Compute layout
    pos: Dict[str, Tuple[float, float]] = {}
    _do_layout(root, 0.0, 0.0, pos)

    # Bounding box
    valid = [n for n in nodes if n.get("name") in pos]
    xs = [pos[n["name"]][0] for n in valid]
    ys = [pos[n["name"]][1] for n in valid]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    tree_w = x_max - x_min + _NW + _H_GAP * 2
    tree_h = y_max - y_min + _NH * 2

    # Scale so each node is at least 1.4 inches wide
    min_node_in = 1.4          # inches
    scale = max(min_node_in / _NW, 12.0 / tree_w)
    if figsize is None:
        fw = min(tree_w * scale, 44.0)
        fh = min(tree_h * scale * 0.7, 28.0)
        figsize = (max(fw, 10.0), max(fh, 6.0))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(x_min - _NW * 0.8, x_max + _NW * 0.8)
    ax.set_ylim(y_min - _NH, y_max + _NH)
    ax.axis("off")

    # ── Edges ─────────────────────────────────────────────────────────────────
    for node in nodes:
        nm = node.get("name")
        if nm not in pos:
            continue
        px, py = pos[nm]
        for child in node.get("children", []):
            cnm = child.get("name")
            if cnm not in pos:
                continue
            cx, cy = pos[cnm]
            ax.annotate(
                "",
                xy=(cx, cy + _NH / 2),
                xytext=(px, py - _NH / 2),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#666666",
                    lw=0.9,
                    mutation_scale=10,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=1,
            )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for node in nodes:
        nm = node.get("name")
        if nm not in pos:
            continue
        x, y = pos[nm]
        _draw_node(ax, node, x, y)

    # ── Colour bar legend ─────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.93, 0.12, 0.016, 0.76])
    norm = plt.Normalize(0.0, 1.0)
    sm = cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P(correct | trajectory)", fontsize=8, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout(rect=[0, 0, 0.92, 1.0])

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
