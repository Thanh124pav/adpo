"""
Full TPO analysis pipeline.

    analyse_tpo(question, gold_answer, server_url, model_name, ...)      [sync]
    analyse_tpo_async(question, gold_answer, server_url, model_name, ...)  [async]

Each function builds a TPO search tree and then annotates every node with:

    name         – hierarchical name ("root", "n1", "n1.2", …)
    V            – mean fraction of correct leaves reachable from this node
    P            – log P(gold_answer | trajectory to this node)  ← set during tree build
    top_logprobs – [(token, log_prob), …]  top-K next-token distribution
    JSD          – {sibling_name: JSD value}  Jensen-Shannon div. between siblings
    correct      – bool (leaves only)

The pipeline is intentionally parallel to ``mc_analysis.analyse`` /
``mc_analysis.analyse_async`` so results are directly comparable and the same
visualisation tools (``mc_analysis.print_tree``, ``mc_analysis.visualise``,
``mc_analysis.visualise_html``) work on TPO trees.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from mc_analysis._annotation import assign_names, compute_v, compute_jsd
from mc_analysis._answer_utils import extract_answer, default_answer_checker
from mc_analysis.vllm_helpers import annotate_top_logprobs, annotate_top_logprobs_async

from .algorithm import build_tree_tpo, build_tree_tpo_async
from .strategies.branching import BranchingStrategy, EntropyBranchingStrategy
from .strategies.termination import TerminationStrategy, PDegradationTerminationStrategy
from .strategies.segmentation import SegmentationStrategy, FixedTokenSegmentation

Node = Dict[str, Any]


def analyse_tpo(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    *,
    branching_strategy: Optional[BranchingStrategy] = None,
    termination_strategy: Optional[TerminationStrategy] = None,
    segmentation_strategy: Optional[SegmentationStrategy] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    top_k_logprobs: int = 20,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> Node:
    """Full TPO analysis pipeline (synchronous).

    Steps
    -----
    1. ``build_tree_tpo``     – build a search tree with dynamic branching and
                                P-degradation termination; P is computed inline.
    2. ``assign_names``       – hierarchical naming ("root", "n1", "n1.2", …).
    3. ``compute_v``          – bottom-up SPO value (mean rollout correctness).
    4. ``annotate_top_logprobs`` – top-K next-token distributions (for JSD).
                                  Nodes populated by entropy branching already
                                  have ``top_logprobs``; remaining nodes are
                                  filled in here.
    5. ``compute_jsd``        – pairwise JSD between sibling next-token
                                distributions.

    Parameters
    ----------
    question : str
        The input prompt / question.
    gold_answer : str
        Reference answer string.  Used both to compute P(answer|trajectory)
        during tree building and to evaluate leaf correctness for V.
    server_url : str
        vLLM server base URL, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model identifier on the vLLM server.
    branching_strategy : BranchingStrategy, optional
        Defaults to :class:`~tpo.strategies.branching.EntropyBranchingStrategy`.
    termination_strategy : TerminationStrategy, optional
        Defaults to :class:`~tpo.strategies.termination.PDegradationTerminationStrategy`
        (K=3, max_depth=8).
    segmentation_strategy : SegmentationStrategy, optional
        Defaults to :class:`~tpo.strategies.segmentation.FixedTokenSegmentation`
        (512 tokens, temperature=0.8).
    answer_checker : callable, optional
        ``checker(generated, gold) → bool``.  Defaults to
        :func:`mc_analysis._answer_utils.default_answer_checker`.
    extract_answer_fn : callable, optional
        ``fn(text) → str | None``.  Defaults to
        :func:`mc_analysis._answer_utils.extract_answer`.
    top_k_logprobs : int
        Number of top tokens to fetch per node for JSD computation.
    seed : int, optional
        RNG seed forwarded to vLLM.
    api_key : str, optional
        Bearer token for authenticated external APIs.
    verbose : bool
        Print phase-level progress to stdout.

    Returns
    -------
    Node
        Annotated root of the TPO search tree.
    """
    checker = answer_checker or default_answer_checker
    ext_fn = extract_answer_fn or extract_answer

    # ---- step 1: build tree (P computed inline) ---------------------------
    root = build_tree_tpo(
        server_url=server_url,
        model_name=model_name,
        prompt=question,
        gold_answer=gold_answer,
        branching_strategy=branching_strategy,
        termination_strategy=termination_strategy,
        segmentation_strategy=segmentation_strategy,
        extract_answer_fn=ext_fn,
        top_k_entropy=top_k_logprobs,
        seed=seed,
        api_key=api_key,
        verbose=verbose,
    )

    # ---- step 2: name nodes -----------------------------------------------
    assign_names(root)

    # ---- step 3: SPO value (mean correctness) ------------------------------
    compute_v(root, gold_answer, checker)

    # ---- step 4: top-K logprobs (fills nodes not already annotated) --------
    annotate_top_logprobs(root, server_url, model_name, top_k=top_k_logprobs, api_key=api_key)

    # ---- step 5: pairwise JSD between siblings ----------------------------
    compute_jsd(root)

    return root


async def analyse_tpo_async(
    question: str,
    gold_answer: str,
    server_url: str,
    model_name: str,
    *,
    branching_strategy: Optional[BranchingStrategy] = None,
    termination_strategy: Optional[TerminationStrategy] = None,
    segmentation_strategy: Optional[SegmentationStrategy] = None,
    answer_checker: Optional[Callable[[Optional[str], str], bool]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    top_k_logprobs: int = 20,
    max_concurrent: int = 32,
    score_concurrent: int = 8,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Node:
    """Full TPO analysis pipeline (asynchronous).

    Tree building and P scoring run concurrently; JSD annotation is
    parallelised after tree construction completes.

    Parameters
    ----------
    max_concurrent : int
        Concurrency for tree-building (generation + scoring).  Default 32.
    score_concurrent : int
        Concurrency for the post-hoc JSD / top-logprob annotation.  Default 8.
    (All other parameters identical to :func:`analyse_tpo`.)
    """
    checker = answer_checker or default_answer_checker
    ext_fn = extract_answer_fn or extract_answer

    t0 = time.perf_counter()
    root = await build_tree_tpo_async(
        server_url=server_url,
        model_name=model_name,
        prompt=question,
        gold_answer=gold_answer,
        branching_strategy=branching_strategy,
        termination_strategy=termination_strategy,
        segmentation_strategy=segmentation_strategy,
        extract_answer_fn=ext_fn,
        top_k_entropy=top_k_logprobs,
        max_concurrent=max_concurrent,
        seed=seed,
        api_key=api_key,
        verbose=verbose,
    )
    if verbose:
        print(f"  [tpo pipeline] tree built in {time.perf_counter() - t0:.1f}s", flush=True)

    assign_names(root)
    compute_v(root, gold_answer, checker)

    t0 = time.perf_counter()
    await annotate_top_logprobs_async(
        root, server_url, model_name,
        top_k=top_k_logprobs,
        max_concurrent=score_concurrent,
        api_key=api_key,
        verbose=verbose,
    )
    if verbose:
        print(f"  [tpo pipeline] JSD done in {time.perf_counter() - t0:.1f}s", flush=True)

    compute_jsd(root)
    return root
