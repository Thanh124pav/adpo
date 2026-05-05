"""
TPO (Tree Path Optimization) tree-building algorithm.

Analogous to ``mc_analysis.vllm_helpers.build_tree`` / ``build_tree_async``
but with two key differences mandated by the TPO specification:

1. **Dynamic branching** — the number of children sampled per node is
   determined by a pluggable :class:`~tpo.strategies.branching.BranchingStrategy`
   (default: :class:`~tpo.strategies.branching.EntropyBranchingStrategy`, which
   reads the next-token entropy and maps it to a branch count in [2, 8]).

2. **P-degradation termination** — in addition to natural EOS / stop-string
   termination, a node is treated as a leaf when a pluggable
   :class:`~tpo.strategies.termination.TerminationStrategy` says so.  The
   default strategy (:class:`~tpo.strategies.termination.PDegradationTerminationStrategy`)
   counts how many nodes along the root-to-node path have a lower
   P(answer|trajectory) than their parent, and terminates when that count
   exceeds a threshold K.

P(answer|trajectory) is computed inline for every node using
:func:`mc_analysis.vllm_helpers.compute_sequence_logprob`, mirroring the
method from ``mc_analysis``.

Public API
----------
build_tree_tpo(server_url, model_name, prompt, gold_answer, ...)  [sync]
build_tree_tpo_async(server_url, model_name, prompt, gold_answer, ...)  [async]
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from mc_analysis.vllm_helpers import (
    _assign_answer,
    _is_terminal_node,
    _make_session,
    _sample_completions,
    compute_sequence_logprob,
    get_next_token_logprobs,
)

from .strategies.branching import BranchingStrategy, EntropyBranchingStrategy
from .strategies.termination import TerminationStrategy, PDegradationTerminationStrategy
from .strategies.segmentation import SegmentationStrategy, FixedTokenSegmentation

Node = Dict[str, Any]


# ---------------------------------------------------------------------------
# Synchronous tree builder
# ---------------------------------------------------------------------------

def build_tree_tpo(
    server_url: str,
    model_name: str,
    prompt: str,
    gold_answer: str,
    branching_strategy: Optional[BranchingStrategy] = None,
    termination_strategy: Optional[TerminationStrategy] = None,
    segmentation_strategy: Optional[SegmentationStrategy] = None,
    *,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    get_logprobs: bool = True,
    top_k_entropy: int = 20,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> Node:
    """Build a TPO search tree by recursively sampling continuations from vLLM.

    Node structure (superset of SPO's / mc_analysis's Node convention)::

        {
            "text":          str,          # text generated at this node
            "full_text":     str,          # prompt + all ancestor texts + this text
            "depth":         int,
            "finish_reason": str,          # "stop" | "length" | …
            "P":             float,        # log P(gold_answer | trajectory to node)
            # ---- present when get_logprobs=True ----
            "sum_logprobs":  float,
            "num_tokens":    int,
            "token_logprobs": List[float],
            "tokens":        List[str],
            # ---- present when entropy branching is used ----
            "top_logprobs":  List[Tuple[str, float]],  # (token, log_prob)
            # ---- present on leaf nodes ----
            "answer":        str,
            # ---- present on non-leaf nodes ----
            "children":      List[Node],
        }

    Parameters
    ----------
    server_url : str
        vLLM server base URL, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model identifier on the vLLM server.
    prompt : str
        Initial prompt / question — becomes the root node's text.
    gold_answer : str
        Reference answer used to compute P(answer|trajectory) at every node.
        Passed to :func:`mc_analysis.vllm_helpers.compute_sequence_logprob`.
    branching_strategy : BranchingStrategy, optional
        Controls how many children to sample per node.
        Defaults to :class:`~tpo.strategies.branching.EntropyBranchingStrategy`
        with ``min_branches=2``, ``max_branches=8``.
    termination_strategy : TerminationStrategy, optional
        Controls when a node is treated as a leaf.
        Defaults to :class:`~tpo.strategies.termination.PDegradationTerminationStrategy`
        with ``max_degradations=3``, ``max_depth=8``.
    segmentation_strategy : SegmentationStrategy, optional
        Controls generation parameters per node (token budget, temperature,
        stop strings).  Defaults to :class:`~tpo.strategies.segmentation.FixedTokenSegmentation`
        with ``max_tokens=512``, ``temperature=0.8``.
    extract_answer_fn : callable, optional
        ``text → answer_str | None``.  Applied to each leaf node.
    get_logprobs : bool
        Whether to request and store per-token log-probs during generation.
    top_k_entropy : int
        Number of top tokens fetched for entropy estimation (only used when
        *branching_strategy* is entropy-based or otherwise requests top-K
        logprobs via :meth:`~tpo.strategies.branching.BranchingStrategy.needs_top_logprobs`).
    seed : int, optional
        RNG seed passed to vLLM.
    api_key : str, optional
        Bearer token for authenticated external APIs.
    verbose : bool
        Print per-node progress to stdout.

    Returns
    -------
    Node
        The root node of the constructed tree, with ``"P"`` set on every node.
    """
    # ---- strategy defaults ------------------------------------------------
    bs = branching_strategy or EntropyBranchingStrategy()
    ts = termination_strategy or PDegradationTerminationStrategy()
    ss = segmentation_strategy or FixedTokenSegmentation()

    session = _make_session(api_key)
    needs_tlp = bs.needs_top_logprobs()
    _expanded = [0]

    # ---- root node --------------------------------------------------------
    root: Node = {"text": prompt, "full_text": prompt, "depth": 0}
    _score_node(root, server_url, model_name, gold_answer)

    def _dfs(node: Node, path: List[Node]) -> None:
        depth = node.get("depth", len(path))
        current_path = path + [node]

        # Fetch top-K next-token logprobs if the branching strategy needs them
        if needs_tlp and "top_logprobs" not in node:
            node["top_logprobs"] = get_next_token_logprobs(
                server_url, model_name, node["full_text"], top_k_entropy, session
            )

        n_branches = bs.get_branch_count(node, depth)
        gen_params = ss.get_generation_params(node, depth)
        stop = gen_params.get("stop")

        children = _sample_completions(
            server_url=server_url,
            model_name=model_name,
            prefix=node["full_text"],
            n=n_branches,
            max_tokens=gen_params["max_tokens"],
            temperature=gen_params["temperature"],
            top_p=gen_params["top_p"],
            stop=stop,
            get_logprobs=get_logprobs,
            seed=seed,
            session=session,
        )

        _expanded[0] += 1
        if verbose:
            print(f"\r  [tpo] {_expanded[0]} nodes expanded, depth={depth}  ", end="", flush=True)

        for child in children:
            child["depth"] = depth + 1
            # Compute P(answer | trajectory to this child) inline
            _score_node(child, server_url, model_name, gold_answer)
            child_path = current_path + [child]

            if _is_terminal_node(child, stop):
                # Natural leaf: EOS or stop string matched but without further expansion
                _assign_answer(child, extract_answer_fn)
            elif ts.should_terminate(child, child_path):
                # Strategic leaf: termination strategy says stop
                _assign_answer(child, extract_answer_fn)
            else:
                _dfs(child, current_path)

        node["children"] = children

    _dfs(root, [])

    if verbose:
        print(f"\r  [tpo] {_expanded[0]} nodes expanded.                ", flush=True)

    return root


# ---------------------------------------------------------------------------
# Asynchronous tree builder
# ---------------------------------------------------------------------------

async def build_tree_tpo_async(
    server_url: str,
    model_name: str,
    prompt: str,
    gold_answer: str,
    branching_strategy: Optional[BranchingStrategy] = None,
    termination_strategy: Optional[TerminationStrategy] = None,
    segmentation_strategy: Optional[SegmentationStrategy] = None,
    *,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    get_logprobs: bool = True,
    top_k_entropy: int = 20,
    max_concurrent: int = 32,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Node:
    """Async version of :func:`build_tree_tpo` with concurrent node expansion.

    Children at each level are expanded concurrently via ``asyncio`` tasks,
    mirroring the design of :func:`mc_analysis.vllm_helpers.build_tree_async`.

    Parameters
    ----------
    max_concurrent : int
        Maximum simultaneous HTTP requests (generation + scoring combined).
    (All other parameters identical to :func:`build_tree_tpo`.)
    """
    bs = branching_strategy or EntropyBranchingStrategy()
    ts = termination_strategy or PDegradationTerminationStrategy()
    ss = segmentation_strategy or FixedTokenSegmentation()

    sem = asyncio.Semaphore(max_concurrent)
    session = _make_session(api_key)
    needs_tlp = bs.needs_top_logprobs()
    _expanded = [0]

    async def _run_in_executor(fn):
        loop = asyncio.get_event_loop()
        async with sem:
            return await loop.run_in_executor(None, fn)

    async def _async_sample(prefix: str, n: int, gen_params: Dict[str, Any]) -> List[Node]:
        return await _run_in_executor(
            lambda: _sample_completions(
                server_url, model_name, prefix, n,
                gen_params["max_tokens"], gen_params["temperature"],
                gen_params["top_p"], gen_params.get("stop"),
                get_logprobs, seed, session,
            )
        )

    async def _async_score(node: Node) -> None:
        result = await _run_in_executor(
            lambda: compute_sequence_logprob(
                server_url, model_name, node["full_text"], gold_answer
            )
        )
        node["P"] = float(result["sum_logprob"])

    async def _async_top_logprobs(node: Node) -> None:
        tlp = await _run_in_executor(
            lambda: get_next_token_logprobs(
                server_url, model_name, node["full_text"], top_k_entropy, session
            )
        )
        node["top_logprobs"] = tlp

    async def _dfs(node: Node, path: List[Node]) -> None:
        depth = node.get("depth", len(path))
        current_path = path + [node]

        # Fetch top-K logprobs for entropy branching (concurrent with nothing else here)
        if needs_tlp and "top_logprobs" not in node:
            await _async_top_logprobs(node)

        n_branches = bs.get_branch_count(node, depth)
        gen_params = ss.get_generation_params(node, depth)
        stop = gen_params.get("stop")

        children = await _async_sample(node["full_text"], n_branches, gen_params)

        _expanded[0] += 1
        if verbose:
            print(f"\r  [tpo] {_expanded[0]} nodes expanded, depth={depth}  ", end="", flush=True)

        # Score all children concurrently (P computation)
        await asyncio.gather(*[_async_score(c) for c in children])

        # Classify children and launch expansion tasks
        expand_tasks = []
        for child in children:
            child["depth"] = depth + 1
            child_path = current_path + [child]

            if _is_terminal_node(child, stop):
                _assign_answer(child, extract_answer_fn)
            elif ts.should_terminate(child, child_path):
                _assign_answer(child, extract_answer_fn)
            else:
                expand_tasks.append(asyncio.create_task(_dfs(child, current_path)))

        if expand_tasks:
            await asyncio.gather(*expand_tasks)

        node["children"] = children

    # ---- root ---------------------------------------------------------------
    root: Node = {"text": prompt, "full_text": prompt, "depth": 0}
    await _async_score(root)

    if verbose:
        print("  [tpo] building tree (async)...", flush=True)

    await _dfs(root, [])

    if verbose:
        print(f"\r  [tpo] {_expanded[0]} nodes expanded.                ", flush=True)

    return root


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_node(
    node: Node,
    server_url: str,
    model_name: str,
    gold_answer: str,
) -> None:
    """Compute P(gold_answer | trajectory to node) and store in node["P"].

    Uses :func:`mc_analysis.vllm_helpers.compute_sequence_logprob` (echo-based
    method): two requests to vLLM to isolate the log-prob of *gold_answer*
    tokens conditioned on ``node["full_text"]``.
    """
    result = compute_sequence_logprob(server_url, model_name, node["full_text"], gold_answer)
    node["P"] = float(result["sum_logprob"])
