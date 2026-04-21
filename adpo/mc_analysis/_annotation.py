"""
Tree annotation: naming, V (SPO value), P (log-prob), JSD (pairwise, named).
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np

Node = Dict[str, Any]
_EPS = 1e-12


# ── node naming ───────────────────────────────────────────────────────────────

def assign_names(node: Node, name: str = "root") -> None:
    """Assign hierarchical name to every node in-place (DFS)."""
    node["name"] = name
    for i, child in enumerate(node.get("children", []), start=1):
        child_name = f"n{i}" if name == "root" else f"{name}.{i}"
        assign_names(child, child_name)


# ── V: SPO value = mean rollout correctness (bottom-up) ──────────────────────

def compute_v(
    node: Node,
    gold_answer: str,
    checker: Callable[[Optional[str], str], bool],
) -> float:
    """
    V(node) = fraction of correct leaf rollouts reachable from this node.

    - Leaf : V = 1.0 if extracted answer is correct, else 0.0
    - Inner: V = mean(V of children)

    Stores result in node["V"] and returns it.
    """
    children = node.get("children", [])
    if not children:
        answer = node.get("answer")
        correct = bool(answer is not None and checker(answer, gold_answer))
        node["correct"] = correct
        node["V"] = float(correct)
        return node["V"]
    child_vs = [compute_v(c, gold_answer, checker) for c in children]
    node["V"] = float(np.mean(child_vs))
    return node["V"]


# ── JSD: pairwise, keyed by sibling name ─────────────────────────────────────

def _binary_jsd(p: float, q: float) -> float:
    """
    Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q).

    JSD = H(M) − ½[H(P) + H(Q)],  M = (P+Q)/2,  H = binary entropy.
    Normalised by ln(2) → result in [0, 1].
    """
    def _h(x: float) -> float:
        x = float(np.clip(x, _EPS, 1.0 - _EPS))
        return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)

    if abs(p - q) < _EPS:
        return 0.0
    m = (p + q) / 2.0
    return float(np.clip((_h(m) - (_h(p) + _h(q)) / 2.0) / np.log(2.0), 0.0, 1.0))


def compute_jsd(node: Node) -> None:
    """
    For every node N with siblings S_1 … S_{K-1}, store:

        N["JSD"] = { S_j["name"]: JSD(V_N, V_{S_j})  for j ≠ i }

    Root has no siblings → root["JSD"] = {}.
    JSD is based on Bernoulli(V) distributions, normalised to [0, 1].
    """
    node.setdefault("JSD", {})          # root: empty dict

    children = node.get("children", [])
    if not children:
        return

    names = [c["name"] for c in children]
    vs    = [c["V"]    for c in children]
    k = len(children)

    for i, child in enumerate(children):
        child["JSD"] = {
            names[j]: _binary_jsd(vs[i], vs[j])
            for j in range(k) if j != i
        }

    for child in children:
        compute_jsd(child)
