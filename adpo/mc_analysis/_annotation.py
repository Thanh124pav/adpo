"""
Tree annotation: naming, V (SPO value), P (log-prob), JSD (pairwise, named).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

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


# ── JSD: pairwise between top-K next-token distributions ─────────────────────

def _topk_jsd(
    dist_i: List[Tuple[str, float]],
    dist_j: List[Tuple[str, float]],
) -> float:
    """
    Jensen-Shannon divergence between two top-K next-token log-prob distributions.

    Each distribution is a list of (token_string, log_prob) pairs.
    Probability mass not covered by the top-K tokens is collected into a
    shared '<UNK>' bucket so the result is a valid JSD in [0, 1]
    (normalised by log 2).
    """
    def to_prob_dict(dist: List[Tuple[str, float]]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for tok, lp in dist:
            d[tok] = float(np.exp(lp))
        total = sum(d.values())
        if total < 1.0 - _EPS:
            d["<UNK>"] = 1.0 - total
        return d

    pi = to_prob_dict(dist_i)
    pj = to_prob_dict(dist_j)
    vocab = sorted(set(pi) | set(pj))

    p = np.array([pi.get(t, 0.0) for t in vocab], dtype=float)
    q = np.array([pj.get(t, 0.0) for t in vocab], dtype=float)

    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2.0

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > _EPS
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd = (_kl(p, m) + _kl(q, m)) / 2.0
    return float(np.clip(jsd / np.log(2.0), 0.0, 1.0))


def compute_jsd(node: Node) -> None:
    """
    For every node N with siblings S_1 … S_{K-1}, store:

        N["JSD"] = { S_j["name"]: JSD(top_logprobs_N, top_logprobs_{S_j})  for j ≠ i }

    JSD is computed between the top-K next-token log-prob distributions stored
    in node["top_logprobs"] (list of (token, log_prob) pairs).  Call
    annotate_top_logprobs / annotate_top_logprobs_hf before this function.

    Root has no siblings → root["JSD"] = {}.
    """
    node.setdefault("JSD", {})

    children = node.get("children", [])
    if not children:
        return

    names = [c["name"] for c in children]
    dists = [c.get("top_logprobs", []) for c in children]
    k = len(children)

    for i, child in enumerate(children):
        child["JSD"] = {
            names[j]: _topk_jsd(dists[i], dists[j])
            for j in range(k) if j != i
        }

    for child in children:
        compute_jsd(child)
