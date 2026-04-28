"""
Full MC-analysis pipeline.

    analyse(question, gold_answer, server_url, model_name, ...)
    analyse_async(question, gold_answer, server_url, model_name, ...)
        vLLM backend (HTTP to running server).

    analyse_hf(question, gold_answer, backend, ...)
    analyse_hf_async(question, gold_answer, backend, ...)
        HuggingFace backend (local model).

Each function returns the annotated tree root Node with the following
fields set on every node:

    name         – hierarchical name ("root", "n1", "n1.2", …)
    V            – mean fraction of correct leaves reachable from this node
    P            – log P(gold_answer | trajectory to this node)  [negative float]
    top_logprobs – [(token, log_prob), …]  top-K next-token distribution
    JSD          – {sibling_name: JSD(top_logprobs_self, top_logprobs_sibling)}
    correct      – bool (leaves only)
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from ._annotation import assign_names, compute_v, compute_jsd
from ._answer_utils import extract_answer, default_answer_checker

Node = Dict[str, Any]


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def _compute_p_tree_vllm(
    root: Node,
    gold_answer: str,
    server_url: str,
    model_name: str,
) -> None:
    from .vllm_helpers import compute_sequence_logprob

    def _recurse(node: Node) -> None:
        result = compute_sequence_logprob(server_url, model_name, node["full_text"], gold_answer)
        node["P"] = float(result["sum_logprob"])
        for child in node.get("children", []):
            _recurse(child)

    _recurse(root)


async def _compute_p_tree_vllm_async(
    root: Node,
    gold_answer: str,
    server_url: str,
    model_name: str,
    max_concurrent: int = 8,
) -> None:
    from .vllm_helpers import compute_sequence_logprob

    sem = asyncio.Semaphore(max_concurrent)

    async def _score(node: Node) -> None:
        async with sem:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: compute_sequence_logprob(
                    server_url, model_name, node["full_text"], gold_answer
                ),
            )
            node["P"] = float(result["sum_logprob"])

    all_nodes: List[Node] = []

    def _collect(n: Node) -> None:
        all_nodes.append(n)
        for c in n.get("children", []):
            _collect(c)

    _collect(root)
    # Shallowest first → maximises prefix-cache hits when --enable-prefix-caching is on
    all_nodes.sort(key=lambda n: len(n.get("full_text", "")))
    await asyncio.gather(*[_score(n) for n in all_nodes])


# ── public vLLM pipeline ───────────────────────────────────────────────────────

def analyse(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    *,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    compute_p: bool = True,
    top_k_logprobs: int = 20,
    max_concurrent: int = 8,
) -> Node:
    """
    Full MC-analysis pipeline using a vLLM server.

    Steps
    -----
    1. build_tree           – generate the search tree via vLLM
    2. assign_names         – give every node a hierarchical name
    3. compute_v            – bottom-up SPO value (mean rollout correctness)
    4. compute_p            – log P(gold_answer | trajectory) for every node
    5. annotate_top_logprobs – top-K next-token distributions for every node
    6. compute_jsd          – pairwise JSD between siblings (token distributions)

    Parameters
    ----------
    question : str
        The input prompt / question.
    gold_answer : str
        Reference answer string.
    server_url : str
        vLLM server base URL, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model ID on the vLLM server.
    tree_kwargs : dict, optional
        Extra kwargs forwarded to :func:`~.vllm_helpers.build_tree`
        (e.g. ``max_depth``, ``branch_factor``, ``temperature``).
    answer_checker : callable, optional
        ``checker(generated, gold) -> bool``.
        Defaults to :func:`~._answer_utils.default_answer_checker`.
    extract_answer_fn : callable, optional
        ``fn(text) -> str | None``.
        Defaults to :func:`~._answer_utils.extract_answer`.
    compute_p : bool
        If False, skip the log-prob scoring step (faster).
    top_k_logprobs : int
        Number of top tokens to fetch per node for JSD computation.
    max_concurrent : int
        Not used in the synchronous path (kept for API symmetry).

    Returns
    -------
    Node
        Annotated tree root.
    """
    from .vllm_helpers import build_tree, annotate_top_logprobs

    checker = answer_checker or default_answer_checker
    ext_fn  = extract_answer_fn or extract_answer
    tkw     = tree_kwargs or {}

    root = build_tree(server_url, model_name, question, extract_answer_fn=ext_fn, **tkw)

    assign_names(root)
    compute_v(root, gold_answer, checker)

    if compute_p:
        _compute_p_tree_vllm(root, gold_answer, server_url, model_name)

    annotate_top_logprobs(root, server_url, model_name, top_k=top_k_logprobs)
    compute_jsd(root)
    return root


async def analyse_async(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    *,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    compute_p: bool = True,
    top_k_logprobs: int = 20,
    max_concurrent: int = 8,
    score_concurrent: int = 4,
) -> Node:
    """Async version of :func:`analyse` — tree-building, P-scoring, and logprob annotation run concurrently.

    Parameters
    ----------
    max_concurrent : int
        Concurrency for tree-building (generation requests). vLLM handles
        these well via continuous batching. Default 8.
    score_concurrent : int
        Concurrency for echo/scoring requests (compute_p, annotate_top_logprobs).
        Each scoring call sends long-prompt echo requests that are heavy on the
        KV cache. Keep this low (2–4) to avoid exhausting the cache and stalling.
        Default 4.
    """
    from .vllm_helpers import build_tree_async, annotate_top_logprobs_async

    checker = answer_checker or default_answer_checker
    ext_fn  = extract_answer_fn or extract_answer
    tkw     = tree_kwargs or {}

    root = await build_tree_async(
        server_url, model_name, question, extract_answer_fn=ext_fn, **tkw
    )

    assign_names(root)
    compute_v(root, gold_answer, checker)

    if compute_p:
        await _compute_p_tree_vllm_async(
            root, gold_answer, server_url, model_name, max_concurrent=score_concurrent
        )

    await annotate_top_logprobs_async(
        root, server_url, model_name, top_k=top_k_logprobs, max_concurrent=score_concurrent
    )
    compute_jsd(root)
    return root


# ── public HF pipeline ────────────────────────────────────────────────────────

def analyse_hf(
    question: str,
    gold_answer: str,
    backend,
    *,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    compute_p: bool = True,
    top_k_logprobs: int = 20,
    max_concurrent: int = 4,
) -> Node:
    """
    Full MC-analysis pipeline using a HuggingFace model.

    Parameters
    ----------
    backend : HFBackend
        Loaded :class:`~.hf_helpers.HFBackend` instance.
    (other parameters identical to :func:`analyse`)
    """
    checker = answer_checker or default_answer_checker
    ext_fn  = extract_answer_fn or extract_answer
    tkw     = tree_kwargs or {}

    root = backend.build_tree(question, extract_answer_fn=ext_fn, **tkw)

    assign_names(root)
    compute_v(root, gold_answer, checker)

    if compute_p:
        backend.compute_p_tree(root, gold_answer)

    backend.annotate_top_logprobs(root, top_k=top_k_logprobs)
    compute_jsd(root)
    return root


async def analyse_hf_async(
    question: str,
    gold_answer: str,
    backend,
    *,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    compute_p: bool = True,
    top_k_logprobs: int = 20,
    max_concurrent: int = 4,
) -> Node:
    """Async version of :func:`analyse_hf`."""
    checker = answer_checker or default_answer_checker
    ext_fn  = extract_answer_fn or extract_answer
    tkw     = tree_kwargs or {}

    root = await backend.build_tree_async(question, extract_answer_fn=ext_fn, **tkw)

    assign_names(root)
    compute_v(root, gold_answer, checker)

    if compute_p:
        await backend.compute_p_tree_async(root, gold_answer, max_concurrent=max_concurrent)

    await backend.annotate_top_logprobs_async(root, top_k=top_k_logprobs, max_concurrent=max_concurrent)
    compute_jsd(root)
    return root
