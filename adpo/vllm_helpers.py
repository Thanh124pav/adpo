"""
vllm_helpers.py

Standalone helper functions for vLLM-based inference.
All functions communicate with a running vLLM server via its OpenAI-compatible REST API
(typically served at http://localhost:<port>/v1).

Public API
----------
rollout_with_alpha(server_url, model_name, prompt, K, ...)
    Generate one sequence and compute, at every token position i:
      • α_i  = sliding-window mean of log-probs over [i, i+K-1]
      • percentile_i = running percentile rank of α_i in {α_0, …, α_i}

compute_sequence_logprob(server_url, model_name, prompt, completion)
    Return per-token log-probs of `completion` conditioned on `prompt`.

build_tree(server_url, model_name, prompt, ...)          [sync]
build_tree_async(server_url, model_name, prompt, ...)    [async]
    Recursively sample continuations from vLLM to build a search tree.
    Standalone replacement for SPO's TreeInferenceStrategy – no guidance
    library, no verl, works with any pretrained model.

Tree utility helpers
--------------------
extract_leaves(root)        → List[Node]   (nodes that have an "answer" key)
extract_all_paths(root)     → List[List[Node]]   (all root-to-leaf paths)
cumulative_logprob(path)    → float        (sum of sum_logprobs along a path)
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import requests

# Type alias matching SPO's convention
Node = Dict[str, Any]


# ---------------------------------------------------------------------------
# 1. rollout_with_alpha
# ---------------------------------------------------------------------------

def rollout_with_alpha(
    server_url: str,
    model_name: str,
    prompt: str,
    K: int = 10,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate one sequence and annotate every token position with α_i and its
    running percentile rank.

    Definitions
    -----------
    Let logprob[i] be the log-probability of the i-th generated token (0-indexed).
    Let n be the total number of generated tokens.

        α_i = mean( logprob[i], logprob[i+1], …, logprob[min(i+K-1, n-1)] )

    The window shrinks near the end of the sequence (fewer than K tokens remain).

        percentile_i = #{j ≤ i : α_j ≤ α_i} / (i+1) × 100

    This is the inclusive percentile rank of α_i within the running prefix
    {α_0, α_1, …, α_i}, expressed as a value in [0, 100].

    Parameters
    ----------
    server_url : str
        Base URL of the vLLM server, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model identifier as registered in the vLLM server.
    prompt : str
        Input text used as the generation prefix.
    K : int
        Sliding-window width (number of tokens averaged for each α_i).
    max_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature (0 = greedy).
    top_p : float
        Nucleus sampling threshold.
    stop : list of str, optional
        Stop sequences passed to vLLM.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``text``          – generated text (str)
        ``tokens``        – list of token strings
        ``logprobs``      – list of per-token log-probs (float; None→0.0)
        ``alpha``         – list of α_i values (float)
        ``percentile``    – list of running percentile ranks (float, 0–100)
        ``finish_reason`` – "stop" | "length" | …
    """
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": 1,  # request the actual token's log-prob at each position
        "n": 1,
    }
    if stop:
        payload["stop"] = stop
    if seed is not None:
        payload["seed"] = seed

    response = requests.post(f"{server_url}/completions", json=payload)
    response.raise_for_status()
    choice = response.json()["choices"][0]

    tokens: List[str] = choice["logprobs"]["tokens"]
    raw_logprobs: List[Optional[float]] = choice["logprobs"]["token_logprobs"]
    # None can appear at position 0 in some vLLM builds when there is no BOS token
    logprobs: List[float] = [lp if lp is not None else 0.0 for lp in raw_logprobs]

    n = len(logprobs)

    # ---- α_i : sliding-window mean ----------------------------------------
    alphas: List[float] = []
    for i in range(n):
        window = logprobs[i : i + K]      # length ≤ K near the end
        alphas.append(float(np.mean(window)))

    # ---- running percentile rank of α_i ------------------------------------
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    percentiles: List[float] = []
    for i in range(n):
        running = alphas_arr[: i + 1]
        # Fraction of values in the running prefix that are ≤ α_i, scaled to 0–100
        rank = float(np.sum(running <= alphas_arr[i])) / (i + 1) * 100.0
        percentiles.append(rank)

    return {
        "text": choice["text"],
        "tokens": tokens,
        "logprobs": logprobs,
        "alpha": alphas,
        "percentile": percentiles,
        "finish_reason": choice.get("finish_reason", "unknown"),
    }


# ---------------------------------------------------------------------------
# 2. compute_sequence_logprob
# ---------------------------------------------------------------------------

def compute_sequence_logprob(
    server_url: str,
    model_name: str,
    prompt: str,
    completion: str,
) -> Dict[str, Any]:
    """Compute the token-level log-probabilities of `completion` given `prompt`.

    Strategy
    --------
    vLLM's ``echo=True`` mode returns log-probs for every token in the *prompt*
    as well as for any newly generated tokens.  We exploit this as follows:

    1. **Request A** – send ``(prompt + completion)`` as the prompt with
       ``echo=True, max_tokens=1``.  The response contains log-probs for all
       tokens in the concatenated string plus one dummy generated token.

    2. **Request B** – send just ``prompt`` with ``echo=True, max_tokens=1``
       to determine exactly how many tokens the prompt occupies after
       tokenisation.

    3. Slice Request A's token list from ``num_prompt_tokens`` to ``-1``
       (excluding the dummy generated token) to isolate the completion tokens
       and their log-probs.

    .. note::
        BPE tokenisation of ``prompt + completion`` may differ from
        tokenising each part independently at their junction point.
        For typical prompt/completion patterns (e.g. a multi-sentence prompt
        followed by a reasoning chain) this difference is negligible.

    Parameters
    ----------
    server_url : str
        Base URL of the vLLM server, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model identifier.
    prompt : str
        The conditioning context.
    completion : str
        The text whose log-probability we want to evaluate.

    Returns
    -------
    dict with keys:
        ``tokens``       – list of token strings for the completion part
        ``logprobs``     – list of per-token log-probs (None→0.0)
        ``sum_logprob``  – sum of log-probs (float)
        ``mean_logprob`` – mean log-prob normalised by token count (float)
        ``num_tokens``   – number of completion tokens (int)
    """
    full_text = prompt + completion

    def _echo_request(text: str) -> Dict[str, Any]:
        r = requests.post(
            f"{server_url}/completions",
            json={
                "model": model_name,
                "prompt": text,
                "max_tokens": 1,      # generate 1 dummy token (discarded)
                "temperature": 0.0,
                "logprobs": 1,
                "echo": True,         # include prompt tokens in the response
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["logprobs"]

    # Request A: full text → logprobs for all tokens (prompt + completion + 1 dummy)
    logprobs_full = _echo_request(full_text)
    all_tokens: List[str] = logprobs_full["tokens"]
    all_logprobs: List[Optional[float]] = logprobs_full["token_logprobs"]

    # Request B: prompt only → count how many tokens the prompt occupies
    # (subtract 1 for the dummy generated token that echo always appends)
    logprobs_prompt = _echo_request(prompt)
    num_prompt_tokens: int = len(logprobs_prompt["tokens"]) - 1

    # Isolate completion tokens: skip prompt tokens and the trailing dummy token
    completion_tokens: List[str] = all_tokens[num_prompt_tokens:-1]
    raw_comp_logprobs: List[Optional[float]] = all_logprobs[num_prompt_tokens:-1]
    completion_logprobs: List[float] = [
        lp if lp is not None else 0.0 for lp in raw_comp_logprobs
    ]

    sum_lp = float(np.sum(completion_logprobs))
    mean_lp = float(np.mean(completion_logprobs)) if completion_logprobs else 0.0

    return {
        "tokens": completion_tokens,
        "logprobs": completion_logprobs,
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "num_tokens": len(completion_tokens),
    }


# ---------------------------------------------------------------------------
# 3. Tree building – shared helpers
# ---------------------------------------------------------------------------

def _resolve_branch_factor(branch_factor: Union[int, List[int]], depth: int) -> int:
    if isinstance(branch_factor, int):
        return branch_factor
    idx = min(depth, len(branch_factor) - 1)
    return branch_factor[idx]


def _sample_completions(
    server_url: str,
    model_name: str,
    prefix: str,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    get_logprobs: bool,
    seed: Optional[int],
) -> List[Node]:
    """Sample *n* completions from vLLM and return them as raw Node dicts."""
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prefix,
        "max_tokens": max_tokens,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": 1 if get_logprobs else 0,
    }
    if stop:
        payload["stop"] = stop
    if seed is not None:
        payload["seed"] = seed

    r = requests.post(f"{server_url}/completions", json=payload)
    r.raise_for_status()

    nodes: List[Node] = []
    for choice in r.json()["choices"]:
        text: str = choice["text"]
        finish_reason: str = choice.get("finish_reason", "unknown")

        node: Node = {
            "text": text,
            "full_text": prefix + text,
            "finish_reason": finish_reason,
        }

        if get_logprobs and choice.get("logprobs"):
            raw_lps: List[Optional[float]] = choice["logprobs"]["token_logprobs"]
            valid_lps = [lp for lp in raw_lps if lp is not None]
            node["sum_logprobs"] = float(np.sum(valid_lps)) if valid_lps else 0.0
            node["num_tokens"] = len(valid_lps)
            node["token_logprobs"] = valid_lps
            node["tokens"] = choice["logprobs"]["tokens"]

        nodes.append(node)

    return nodes


def _is_terminal_node(node: Node, stop: Optional[List[str]]) -> bool:
    """Return True when a node should not be expanded further.

    A node is terminal if:
    • The model was truncated (finish_reason == "length"), or
    • Stop sequences were NOT provided (no intermediate stops defined), or
    • The generated text does NOT end with any of the defined stop sequences
      (meaning the model stopped naturally → treat as leaf).
    """
    if node["finish_reason"] == "length":
        return True
    if stop and any(node["text"].endswith(s) for s in stop):
        # Stopped on an intermediate sequence → continue expanding
        return False
    return True  # stopped naturally or no stop sequences → leaf


def _assign_answer(
    node: Node,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]],
) -> None:
    if extract_answer_fn is not None:
        answer = extract_answer_fn(node["text"])
        if answer is not None:
            node["answer"] = answer
    else:
        node["answer"] = node["text"]


# ---------------------------------------------------------------------------
# 3a. build_tree  (synchronous)
# ---------------------------------------------------------------------------

def build_tree(
    server_url: str,
    model_name: str,
    prompt: str,
    max_depth: int = 2,
    branch_factor: Union[int, List[int]] = 3,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    get_logprobs: bool = True,
    seed: Optional[int] = None,
) -> Node:
    """Build a search tree by recursively sampling continuations from vLLM.

    This is a **standalone** replacement for SPO's ``TreeInferenceStrategy``.
    It has no dependency on the guidance library, verl, or the training stack –
    just a running vLLM server.

    Tree node structure (compatible with SPO's Node convention)::

        {
            "text":          str,         # text generated at this node
            "full_text":     str,         # prompt + all ancestor texts + this text
            "depth":         int,
            "finish_reason": str,         # "stop" | "length" | …
            # ---- present when get_logprobs=True ----
            "sum_logprobs":  float,       # sum of token log-probs for this node
            "num_tokens":    int,
            "token_logprobs": List[float],
            "tokens":        List[str],
            # ---- present on leaf nodes ----
            "answer":        str,         # extracted or raw text
            # ---- present on non-leaf nodes ----
            "children":      List[Node],
        }

    Parameters
    ----------
    server_url : str
        Base URL, e.g. ``"http://localhost:8000/v1"``.
    model_name : str
        Model identifier on the vLLM server.
    prompt : str
        Initial prompt – this becomes the root node's text.
    max_depth : int
        Maximum expansion depth.  The root is at depth 0; its children at
        depth 1; etc.  Nodes at depth ``max_depth`` are not expanded further.
    branch_factor : int or List[int]
        Number of children sampled per node.  If a list is given,
        ``branch_factor[d]`` is used at depth ``d``; the last element is
        reused for any deeper level.
    max_tokens : int
        Maximum tokens to generate per node.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling threshold.
    stop : list of str, optional
        Intermediate stop sequences.  When a generated continuation ends
        with one of these strings the node is treated as *intermediate* and
        expanded further.  Nodes that stop for any other reason (natural stop,
        length truncation) are treated as *leaves*.
        If ``None``, every child is immediately a leaf.
    extract_answer_fn : callable, optional
        ``text → answer_str | None``.  Applied to each leaf node's text.
        When it returns ``None`` the node is left without an ``"answer"`` key.
        Default: the raw text is used as the answer.
    get_logprobs : bool
        Whether to request and store per-token log-probs.
    seed : int, optional
        RNG seed passed to vLLM.

    Returns
    -------
    Node
        The root node of the constructed tree.
    """
    def _dfs(node: Node, depth: int) -> None:
        if depth >= max_depth:
            _assign_answer(node, extract_answer_fn)
            return

        bf = _resolve_branch_factor(branch_factor, depth)
        children = _sample_completions(
            server_url=server_url,
            model_name=model_name,
            prefix=node["full_text"],
            n=bf,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            get_logprobs=get_logprobs,
            seed=seed,
        )

        for child in children:
            child["depth"] = depth + 1
            if _is_terminal_node(child, stop):
                _assign_answer(child, extract_answer_fn)
            else:
                _dfs(child, depth + 1)

        node["children"] = children

    root: Node = {
        "text": prompt,
        "full_text": prompt,
        "depth": 0,
    }
    _dfs(root, 0)
    return root


# ---------------------------------------------------------------------------
# 3b. build_tree_async  (asynchronous, concurrent node expansion)
# ---------------------------------------------------------------------------

async def build_tree_async(
    server_url: str,
    model_name: str,
    prompt: str,
    max_depth: int = 2,
    branch_factor: Union[int, List[int]] = 3,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    get_logprobs: bool = True,
    seed: Optional[int] = None,
    max_concurrent: int = 64,
) -> Node:
    """Async version of :func:`build_tree` with concurrent node expansion.

    Children at each level are expanded concurrently using ``asyncio`` tasks,
    mirroring the design of SPO's ``TreeInferenceStrategy._construct_tree``.

    All parameters are identical to :func:`build_tree` except:

    Parameters
    ----------
    max_concurrent : int
        Maximum number of simultaneous HTTP requests to the vLLM server.

    Returns
    -------
    Node
        The root node of the constructed tree.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _async_sample(prefix: str, n: int) -> List[Node]:
        async with sem:
            loop = asyncio.get_event_loop()
            # Run the blocking HTTP call in a thread-pool executor so we don't
            # block the event loop.
            return await loop.run_in_executor(
                None,
                lambda: _sample_completions(
                    server_url, model_name, prefix, n,
                    max_tokens, temperature, top_p, stop,
                    get_logprobs, seed,
                ),
            )

    async def _dfs(node: Node, depth: int) -> None:
        if depth >= max_depth:
            _assign_answer(node, extract_answer_fn)
            return

        bf = _resolve_branch_factor(branch_factor, depth)
        children = await _async_sample(node["full_text"], bf)

        expand_tasks = []
        for child in children:
            child["depth"] = depth + 1
            if _is_terminal_node(child, stop):
                _assign_answer(child, extract_answer_fn)
            else:
                expand_tasks.append(asyncio.create_task(_dfs(child, depth + 1)))

        if expand_tasks:
            await asyncio.gather(*expand_tasks)

        node["children"] = children

    root: Node = {
        "text": prompt,
        "full_text": prompt,
        "depth": 0,
    }
    await _dfs(root, 0)
    return root


# ---------------------------------------------------------------------------
# Tree utility helpers
# ---------------------------------------------------------------------------

def extract_leaves(root: Node) -> List[Node]:
    """Return all leaf nodes (nodes that carry an ``"answer"`` key).

    Parameters
    ----------
    root : Node
        Root of the tree returned by :func:`build_tree`.

    Returns
    -------
    List[Node]
        All leaf nodes in DFS order.
    """
    leaves: List[Node] = []

    def _dfs(node: Node) -> None:
        if "answer" in node:
            leaves.append(node)
        for child in node.get("children", []):
            _dfs(child)

    _dfs(root)
    return leaves


def extract_all_paths(root: Node) -> List[List[Node]]:
    """Return every root-to-leaf path as a list of nodes.

    Each path is ordered from root (depth 0) to leaf (deepest node with
    an ``"answer"`` key).

    Parameters
    ----------
    root : Node
        Root of the tree returned by :func:`build_tree`.

    Returns
    -------
    List[List[Node]]
        Each inner list is one complete path.
    """
    paths: List[List[Node]] = []

    def _dfs(node: Node, current_path: List[Node]) -> None:
        current_path = current_path + [node]
        if "answer" in node:
            paths.append(current_path)
        for child in node.get("children", []):
            _dfs(child, current_path)

    _dfs(root, [])
    return paths


def cumulative_logprob(path: List[Node]) -> float:
    """Compute the cumulative (summed) log-probability of a root-to-leaf path.

    Equivalent to ``sum(node["sum_logprobs"] for node in path if "sum_logprobs" in node)``.

    Parameters
    ----------
    path : List[Node]
        A path as returned by :func:`extract_all_paths`.

    Returns
    -------
    float
        Sum of ``sum_logprobs`` across all nodes in the path.
    """
    return sum(node.get("sum_logprobs", 0.0) for node in path)
