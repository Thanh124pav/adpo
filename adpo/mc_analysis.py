"""
mc_analysis.py

Monte Carlo tree analysis pipeline for SPO generation trees.

Pipeline
--------
Given a (question, gold_answer) pair:

1. **Build tree** — generate continuations from vLLM using the SPO build_tree
   algorithm (standalone, no guidance/verl dependency).

2. **Compute V** — SPO value at every node (same definition as original SPO):
       V(node) = mean rollout correctness of all leaves under this node
   - Leaf: V = 1.0 if extracted answer matches gold_answer, else 0.0
   - Internal node: V = mean(V of children)

3. **Compute P** — language-model probability of the gold answer given the
   trajectory that leads to this node:
       P(node) = exp( mean_log_prob( gold_answer | full_trajectory_to_node ) )
   Computed via a single ``/v1/completions`` call per node using
   ``vllm_helpers.compute_sequence_logprob``.

4. **Compute JSD** — diversity of the sibling group.  For each set of siblings
   {S_0, …, S_{k-1}} (children of the same parent), compute the generalised
   Jensen-Shannon divergence of the group (a single real number shared by
   every sibling in that group):
       JSD_group = H( mean_V ) - mean( H(V_i) )    (normalised to [0, 1])
   where H is binary entropy and V_i = V(S_i).
   Root node (no siblings) receives JSD = 0.0.

5. **Visualise** — draw the annotated tree with matplotlib.  Every node box
   shows: name, truncated text, V, P, JSD.  Node colour encodes V via a
   Red→Yellow→Green colourmap.

Public API
----------
    analyse(question, gold_answer, server_url, model_name, ...) → Node
    analyse_async(...)                                           → Node  (async)
    print_tree(root)                                            (debug text dump)
    visualise(root, output_path=None, ...)                      → Figure
"""

import asyncio
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """Extract the final answer from generated text (\\boxed{…} or last line)."""
    if _ext_boxed is not None:
        ans = _ext_boxed(text)
        if ans is not None:
            return ans
    import re
    m = re.search(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def default_answer_checker(generated: Optional[str], gold: str) -> bool:
    """Equivalence check using adpo.reward_functions.is_equiv when available."""
    gen = (generated or "").strip()
    ref = (gold or "").strip()
    if _is_equiv is not None:
        return bool(_is_equiv(gen, ref))
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
# Step 1 — node naming
# =============================================================================

def _assign_names(node: Node, name: str = "root") -> None:
    """Assign a hierarchical name to every node in-place."""
    node["name"] = name
    for i, child in enumerate(node.get("children", []), start=1):
        child_name = f"n{i}" if name == "root" else f"{name}.{i}"
        _assign_names(child, child_name)


# =============================================================================
# Step 2 — V: SPO value = mean rollout correctness  (bottom-up)
# =============================================================================

def _compute_v(
    node: Node,
    gold_answer: str,
    checker: Callable[[Optional[str], str], bool],
) -> float:
    """Compute and store V on every node.  Returns node["V"].

    V is the original SPO value: the fraction of correct leaf rollouts
    reachable from this node.

    - Leaf node    : V = 1.0 if answer is correct, else 0.0
    - Internal node: V = mean(V of children)
    """
    children = node.get("children", [])
    if not children:
        answer = node.get("answer")
        is_correct = bool(answer is not None and checker(answer, gold_answer))
        node["correct"] = is_correct
        node["V"] = float(is_correct)
        return node["V"]

    child_vs = [_compute_v(c, gold_answer, checker) for c in children]
    node["V"] = float(np.mean(child_vs))
    return node["V"]


# =============================================================================
# Step 3 — P: LM probability of gold answer given trajectory  (per-node HTTP)
# =============================================================================

def _compute_p_node(
    node: Node,
    gold_answer: str,
    server_url: str,
    model_name: str,
) -> None:
    """Compute P for a single node via compute_sequence_logprob.

    P(node) = exp( mean_log_prob(gold_answer | full_trajectory_to_node) )

    This is the per-token geometric-mean probability of the gold answer string
    given everything the model has generated up to and including this node.
    It is a real number in (0, 1].
    """
    from adpo.vllm_helpers import compute_sequence_logprob

    result = compute_sequence_logprob(
        server_url=server_url,
        model_name=model_name,
        prompt=node["full_text"],
        completion=gold_answer,
    )
    node["P"] = float(np.exp(result["mean_logprob"]))


def _compute_p_tree(
    node: Node,
    gold_answer: str,
    server_url: str,
    model_name: str,
) -> None:
    """Recursively compute P for every node in the tree (sequential)."""
    _compute_p_node(node, gold_answer, server_url, model_name)
    for child in node.get("children", []):
        _compute_p_tree(child, gold_answer, server_url, model_name)


async def _compute_p_tree_async(
    node: Node,
    gold_answer: str,
    server_url: str,
    model_name: str,
    sem: asyncio.Semaphore,
) -> None:
    """Compute P for every node concurrently (async)."""
    from adpo.vllm_helpers import compute_sequence_logprob

    async def _score(n: Node) -> None:
        async with sem:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: compute_sequence_logprob(
                    server_url=server_url,
                    model_name=model_name,
                    prompt=n["full_text"],
                    completion=gold_answer,
                ),
            )
            n["P"] = float(np.exp(result["mean_logprob"]))

    all_nodes: List[Node] = []

    def _collect(n: Node) -> None:
        all_nodes.append(n)
        for c in n.get("children", []):
            _collect(c)

    _collect(node)
    await asyncio.gather(*[_score(n) for n in all_nodes])


# =============================================================================
# Step 4 — JSD: diversity of sibling group  (single real number per group)
# =============================================================================

def _binary_entropy(x: float) -> float:
    """Binary entropy H(x) = -x log x - (1-x) log(1-x), in nats."""
    x = float(np.clip(x, _EPS, 1.0 - _EPS))
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def multi_bernoulli_jsd(vs: List[float]) -> float:
    """Generalised Jensen-Shannon divergence for a group of Bernoulli(V_i).

    JSD_group = H( mean(V_i) ) - mean( H(V_i) )

    Normalised by ln(2) so the result lies in [0, 1]:
    - 0   → all siblings have identical V  (no diversity)
    - 1   → maximally diverse (half correct, half incorrect)

    Parameters
    ----------
    vs : list of float
        V values of the siblings (each V ∈ [0, 1]).

    Returns
    -------
    float in [0, 1]
    """
    if len(vs) <= 1:
        return 0.0
    mean_v = float(np.mean(vs))
    jsd_nats = _binary_entropy(mean_v) - float(np.mean([_binary_entropy(v) for v in vs]))
    return float(np.clip(jsd_nats / np.log(2.0), 0.0, 1.0))


def _compute_jsd(node: Node) -> None:
    """Assign JSD to every node (in-place, single real number per node).

    All siblings in the same group receive the same JSD value (it is a
    property of the group, not of an individual node).
    Root has no siblings → JSD = 0.0.
    """
    node.setdefault("JSD", 0.0)   # root

    children = node.get("children", [])
    if not children:
        return

    vs = [c["V"] for c in children]
    jsd_val = multi_bernoulli_jsd(vs)
    for child in children:
        child["JSD"] = jsd_val

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
        Defaults to :func:`default_answer_checker`.
    answer_extractor : callable, optional
        ``(text: str) → str | None`` applied by build_tree to each leaf.
        Defaults to :func:`extract_answer`.

    Returns
    -------
    Node
        Root of the annotated tree.  Every node has:
        ``name``, ``V``, ``P``, ``JSD``  (and ``correct`` for leaf nodes).
    """
    from adpo.vllm_helpers import build_tree

    checker = answer_checker or default_answer_checker
    extractor = answer_extractor or extract_answer
    kw: Dict[str, Any] = dict(tree_kwargs or {})

    # 1. Build tree
    root = build_tree(
        server_url=server_url,
        model_name=model_name,
        prompt=question,
        extract_answer_fn=extractor,
        **kw,
    )

    # 2. Names
    _assign_names(root)

    # 3. V — SPO value (no HTTP needed, uses the tree structure)
    _compute_v(root, gold_answer, checker)

    # 4. P — LM probability of gold answer given each trajectory
    _compute_p_tree(root, gold_answer, server_url, model_name)

    # 5. JSD — diversity of sibling groups
    _compute_jsd(root)

    return root


# =============================================================================
# Public: analyse_async  (concurrent tree build + concurrent P scoring)
# =============================================================================

async def analyse_async(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    answer_extractor: Optional[Callable[[str], Optional[str]]] = None,
    max_concurrent_p: int = 32,
) -> Node:
    """Async version of :func:`analyse`.

    The tree is built concurrently (via :func:`build_tree_async`) and P
    is scored for all nodes concurrently (up to *max_concurrent_p*
    simultaneous HTTP requests).

    Extra parameter
    ---------------
    max_concurrent_p : int
        Maximum parallel ``/v1/completions`` calls for P scoring.
    """
    from adpo.vllm_helpers import build_tree_async

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
    _compute_v(root, gold_answer, checker)

    sem = asyncio.Semaphore(max_concurrent_p)
    await _compute_p_tree_async(root, gold_answer, server_url, model_name, sem)

    _compute_jsd(root)

    return root


# =============================================================================
# Text display  (debug)
# =============================================================================

def print_tree(node: Node, indent: int = 0, max_text: int = 60) -> None:
    """Print a compact text representation of the annotated tree.

    Example::

        [root]      V=0.667  P=0.031  JSD=0.000
          text: "What is 2+2?"
          [n1] ✓   V=1.000  P=0.842  JSD=0.918
            text: "2+2=4, so the answer is \\boxed{4}."
          [n2] ✗   V=0.000  P=0.003  JSD=0.918
            text: "I think it's \\boxed{5}."
    """
    name = node.get("name", "?")
    v = node.get("V", 0.0)
    p = node.get("P", 0.0)
    jsd = node.get("JSD", 0.0)

    raw = node.get("text", "")
    disp = raw[:max_text].replace("\n", " ")
    if len(raw) > max_text:
        disp += "…"

    is_leaf = "answer" in node
    mark = (" ✓" if node.get("correct") else " ✗") if is_leaf else ""

    pad = "  " * indent
    print(f"{pad}[{name}]{mark:<3}  V={v:.3f}  P={p:.4f}  JSD={jsd:.3f}")
    print(f'{pad}  text: "{disp}"')

    for child in node.get("children", []):
        print_tree(child, indent + 1, max_text)


# =============================================================================
# Visualisation
# =============================================================================

# Layout constants (data-coordinate units)
_NW    = 3.6    # node box width
_NH    = 2.4    # node box height
_H_GAP = 0.5    # horizontal gap between sibling subtrees
_V_GAP = 3.4    # vertical distance between depth levels
_FS    = 7      # base font size (pt)
_MAX_T = 38     # max chars per display line inside a node box


def _subtree_w(node: Node) -> float:
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
    children = node.get("children", [])
    sw = _subtree_w(node)
    pos[node["name"]] = (x_left + sw / 2.0, y)
    cur = x_left
    for child in children:
        _do_layout(child, cur, y - _V_GAP, pos)
        cur += _subtree_w(child)


def _v_to_colour(v: float):
    """Map V ∈ [0,1] to RGBA via the RdYlGn colourmap."""
    from matplotlib.cm import RdYlGn
    return RdYlGn(float(np.clip(v, 0.0, 1.0)))


def _fmt_text(raw: str, max_chars: int = _MAX_T, max_lines: int = 2) -> str:
    raw = " ".join(raw.split())
    if len(raw) <= max_chars:
        return raw
    return "\n".join(textwrap.wrap(raw, width=max_chars,
                                   max_lines=max_lines, placeholder="…"))


def _draw_node(ax, node: Node, x: float, y: float) -> None:
    import matplotlib.patches as mpatches

    v    = node.get("V", 0.0)
    p    = node.get("P", 0.0)
    jsd  = node.get("JSD", 0.0)
    name = node.get("name", "?")
    is_leaf = "answer" in node

    text_disp = _fmt_text(node.get("text", ""))

    fc = _v_to_colour(v)
    ec = tuple(max(0.0, c - 0.22) for c in fc[:3]) + (1.0,)

    # ── background box ────────────────────────────────────────────────────────
    box = mpatches.FancyBboxPatch(
        (x - _NW / 2, y - _NH / 2), _NW, _NH,
        boxstyle="round,pad=0.07",
        facecolor=fc, edgecolor=ec,
        linewidth=1.3, zorder=2,
    )
    ax.add_patch(box)

    # ── header: name  +  correctness mark (leaves only) ──────────────────────
    ax.text(x, y + _NH / 2 - 0.22, name,
            ha="center", va="top",
            fontsize=_FS + 1.5, fontweight="bold", color="black", zorder=3)

    if is_leaf:
        mark      = " ✓" if node.get("correct") else " ✗"
        mark_col  = "#006600" if node.get("correct") else "#990000"
        ax.text(x + _NW / 2 - 0.12, y + _NH / 2 - 0.22, mark,
                ha="right", va="top",
                fontsize=_FS + 1, fontweight="bold", color=mark_col, zorder=3)

    # ── separator ─────────────────────────────────────────────────────────────
    sep_y = y + _NH / 2 - 0.46
    ax.plot([x - _NW / 2 + 0.08, x + _NW / 2 - 0.08],
            [sep_y, sep_y],
            color="black", lw=0.5, alpha=0.45, zorder=3)

    # ── truncated text ────────────────────────────────────────────────────────
    ax.text(x, sep_y - 0.08,
            f'"{text_disp}"',
            ha="center", va="top",
            fontsize=_FS - 1, style="italic", color="#2a2a2a", zorder=3)

    # ── V, P, JSD ─────────────────────────────────────────────────────────────
    stats_y = y - _NH / 2 + 0.82
    ax.text(x, stats_y,
            f"V: {v:.3f}   P: {p:.4f}   JSD: {jsd:.3f}",
            ha="center", va="top",
            fontsize=_FS, family="monospace", color="black", zorder=3)


def _all_nodes(root: Node) -> List[Node]:
    out = [root]
    for c in root.get("children", []):
        out.extend(_all_nodes(c))
    return out


def visualise(
    root: Node,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    title: str = "SPO Generation Tree  –  MC Analysis",
) -> "plt.Figure":  # type: ignore[name-defined]
    """Draw the annotated SPO tree.

    Each node box contains
    ----------------------
    * **name**  — hierarchical identifier (root, n1, n1.2, …)
    * **text**  — truncated generated continuation (italic)
    * **V**     — SPO value = mean rollout correctness ∈ [0, 1]
    * **P**     — P(answer | trajectory) = exp(mean_logprob) ∈ (0, 1]
    * **JSD**   — sibling-group diversity = generalised JS divergence ∈ [0, 1]

    Node colour encodes **V** via the Red → Yellow → Green colourmap.
    Leaf nodes show ✓ / ✗.

    Parameters
    ----------
    root : Node
        Annotated tree from :func:`analyse` or :func:`analyse_async`.
    output_path : str, optional
        Save to this file path if given (.png / .pdf / .svg / …).
    figsize : (float, float), optional
        Override figure size (width, height) in inches.
    dpi : int
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    nodes = _all_nodes(root)

    pos: Dict[str, Tuple[float, float]] = {}
    _do_layout(root, 0.0, 0.0, pos)

    valid = [n for n in nodes if n.get("name") in pos]
    xs = [pos[n["name"]][0] for n in valid]
    ys = [pos[n["name"]][1] for n in valid]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    tree_w = x_max - x_min + _NW + _H_GAP * 2
    tree_h = y_max - y_min + _NH * 2

    min_node_in = 1.4
    scale = max(min_node_in / _NW, 12.0 / tree_w)
    if figsize is None:
        figsize = (
            max(min(tree_w * scale, 44.0), 10.0),
            max(min(tree_h * scale * 0.7, 28.0), 6.0),
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(x_min - _NW * 0.8, x_max + _NW * 0.8)
    ax.set_ylim(y_min - _NH, y_max + _NH)
    ax.axis("off")

    # ── edges ─────────────────────────────────────────────────────────────────
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
                    arrowstyle="-|>", color="#666666",
                    lw=0.9, mutation_scale=10,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=1,
            )

    # ── nodes ─────────────────────────────────────────────────────────────────
    for node in nodes:
        nm = node.get("name")
        if nm not in pos:
            continue
        x, y = pos[nm]
        _draw_node(ax, node, x, y)

    # ── colourbar  (V) ────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.93, 0.12, 0.016, 0.76])
    norm = plt.Normalize(0.0, 1.0)
    sm = cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("V  (mean rollout correctness)", fontsize=8, labelpad=6)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout(rect=[0, 0, 0.92, 1.0])

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
