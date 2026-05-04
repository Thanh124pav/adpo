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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

# Type alias matching SPO's convention
Node = Dict[str, Any]


def _make_session(api_key: Optional[str] = None) -> requests.Session:
    """Return a requests.Session, optionally pre-configured with Bearer auth.

    Used when querying an external OpenAI-compatible API endpoint instead of
    a locally hosted vLLM server.
    """
    s = requests.Session()
    if api_key:
        s.headers["Authorization"] = f"Bearer {api_key}"
    return s


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

    response = requests.post(f"{server_url}/completions", json=payload, timeout=300)
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
            timeout=300,
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
    session: Optional[requests.Session] = None,
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

    poster = session or requests
    r = poster.post(f"{server_url}/completions", json=payload, timeout=300)
    r.raise_for_status()

    nodes: List[Node] = []
    for choice in r.json()["choices"]:
        text: str = choice["text"]
        finish_reason: str = choice.get("finish_reason", "unknown")

        node: Node = {
            "text": text,
            "full_text": prefix + text,
            "finish_reason": finish_reason,
            # stop_reason: the actual stop string matched, or None for EOS/length.
            # vLLM sets this field; it's None when the model stopped naturally.
            "stop_reason": choice.get("stop_reason"),
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

    Two splitting modes
    ------------------
    M-token mode  (stop=None, default — same as SPO):
        max_tokens is the step size.  A node is expanded further unless the
        model produced a natural EOS before hitting max_tokens.
        • finish_reason == "length"  → hit M tokens = step boundary → NOT terminal
        • finish_reason == "stop", stop_reason is None → natural EOS → terminal

    Stop-sequence mode  (stop=[...]):
        A node is a step boundary only when it ends with a stop string.
        • stop_reason is not None  → hit stop string → NOT terminal
        • stop_reason is None      → natural EOS → terminal
        • finish_reason == "length" → truncated mid-step → terminal
    """
    if node["finish_reason"] == "stop":
        # Natural EOS (stop_reason is None) → always a leaf
        # Hit a stop string (stop_reason is not None) → step boundary → expand
        return node.get("stop_reason") is None

    if node["finish_reason"] == "length":
        # M-token mode: length = step complete → expand further
        # Stop-seq mode: length = truncated mid-step → treat as leaf
        return bool(stop)

    return True  # unknown finish_reason → safe default


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


def _is_short_answer(text: str) -> bool:
    """Return True for bare short answers that look like a single value (no whitespace)."""
    stripped = text.strip()
    return bool(stripped) and len(stripped.split()) == 1


def _normalize_p_answer(raw_answer: str, is_internal: bool) -> str:
    """Normalize an extracted answer string for the inline-P answer branch.

    - Internal nodes: prefix with '...' so the model treats it as a continuation
    - Short answers (single token): wrap in LaTeX boxed format for cleaner display
    """
    raw_answer = raw_answer.strip()
    if is_internal:
        if _is_short_answer(raw_answer):
            return f"...The final answer is \\boxed{{{raw_answer}}}"
        return "..." + raw_answer
    if _is_short_answer(raw_answer):
        return f"The final answer is \\boxed{{{raw_answer}}}"
    return raw_answer


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
    api_key: Optional[str] = None,
    compute_p_inline: bool = False,
    p_max_tokens: int = 4096,
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
    compute_p_inline : bool
        If True, for every expanding node fire one extra "answer branch"
        request (no stop constraints, up to ``p_max_tokens``).  The
        sum-logprob of that branch is stored as ``node["P"]``, replacing
        the post-hoc echo scoring.  Terminal leaf nodes get
        ``P = sum_logprobs`` from their own generation.
    p_max_tokens : int
        Max tokens for the inline answer branch (default 4096).

    Returns
    -------
    Node
        The root node of the constructed tree.
    """
    session = _make_session(api_key)

    def _dfs(node: Node, depth: int) -> None:
        if depth >= max_depth:
            _assign_answer(node, extract_answer_fn)
            if compute_p_inline:
                node["P"] = node.get("sum_logprobs", 0.0)
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
            session=session,
        )

        if compute_p_inline:
            ans_nodes = _sample_completions(
                server_url=server_url,
                model_name=model_name,
                prefix=node["full_text"],
                n=1,
                max_tokens=p_max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=None,
                get_logprobs=True,
                seed=seed,
                session=session,
            )
            ans_branch = ans_nodes[0]
            node["P"] = ans_branch.get("sum_logprobs", 0.0)
            raw_ans = extract_answer_fn(ans_branch["text"]) if extract_answer_fn else ans_branch["text"]
            if raw_ans is not None:
                node["p_answer"] = _normalize_p_answer(raw_ans, is_internal=True)

        for child in children:
            child["depth"] = depth + 1
            if _is_terminal_node(child, stop):
                _assign_answer(child, extract_answer_fn)
                if compute_p_inline:
                    child["P"] = child.get("sum_logprobs", 0.0)
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
    api_key: Optional[str] = None,
    compute_p_inline: bool = False,
    p_max_tokens: int = 4096,
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
    session = _make_session(api_key)

    async def _async_sample(prefix: str, n: int) -> List[Node]:
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: _sample_completions(
                    server_url, model_name, prefix, n,
                    max_tokens, temperature, top_p, stop,
                    get_logprobs, seed, session,
                ),
            )

    async def _async_ans_branch(prefix: str) -> List[Node]:
        """Sample one free-form answer branch (no stop, p_max_tokens) for inline P."""
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: _sample_completions(
                    server_url, model_name, prefix, 1,
                    p_max_tokens, temperature, top_p, None,
                    True, seed, session,
                ),
            )

    async def _dfs(node: Node, depth: int) -> None:
        if depth >= max_depth:
            _assign_answer(node, extract_answer_fn)
            if compute_p_inline:
                node["P"] = node.get("sum_logprobs", 0.0)
            return

        bf = _resolve_branch_factor(branch_factor, depth)

        if compute_p_inline:
            children, ans_nodes = await asyncio.gather(
                _async_sample(node["full_text"], bf),
                _async_ans_branch(node["full_text"]),
            )
            ans_branch = ans_nodes[0]
            node["P"] = ans_branch.get("sum_logprobs", 0.0)
            raw_ans = extract_answer_fn(ans_branch["text"]) if extract_answer_fn else ans_branch["text"]
            if raw_ans is not None:
                node["p_answer"] = _normalize_p_answer(raw_ans, is_internal=True)
        else:
            children = await _async_sample(node["full_text"], bf)

        expand_tasks = []
        for child in children:
            child["depth"] = depth + 1
            if _is_terminal_node(child, stop):
                _assign_answer(child, extract_answer_fn)
                if compute_p_inline:
                    child["P"] = child.get("sum_logprobs", 0.0)
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


# ---------------------------------------------------------------------------
# 5. Top-K next-token log-probs  (used for JSD between siblings)
# ---------------------------------------------------------------------------

def get_next_token_logprobs(
    server_url: str,
    model_name: str,
    text: str,
    top_k: int = 20,
    session: Optional[requests.Session] = None,
) -> List[Tuple[str, float]]:
    """
    Query the top-K next-token log-probs from vLLM at the end of *text*.

    Uses ``max_tokens=1`` with ``logprobs=top_k`` so vLLM returns the
    full top-K distribution without generating meaningful output.

    Returns
    -------
    List[Tuple[str, float]]
        ``[(token_string, log_prob), ...]`` sorted by descending log-prob.
    """
    poster = session or requests
    r = poster.post(
        f"{server_url}/completions",
        json={
            "model": model_name,
            "prompt": text,
            "max_tokens": 1,
            "logprobs": top_k,
            "temperature": 0.0,
        },
        timeout=300,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    # top_logprobs[0] is the distribution at the *first* (and only) generated position,
    # which equals the model's distribution at the end of `text`.
    top_lps: Dict[str, float] = choice["logprobs"]["top_logprobs"][0]
    return sorted(top_lps.items(), key=lambda kv: kv[1], reverse=True)


def annotate_top_logprobs(
    root: Node,
    server_url: str,
    model_name: str,
    top_k: int = 20,
    api_key: Optional[str] = None,
) -> None:
    """Store top-K next-token log-probs in ``node["top_logprobs"]`` for every node (sequential DFS)."""
    session = _make_session(api_key)

    def _recurse(node: Node) -> None:
        node["top_logprobs"] = get_next_token_logprobs(
            server_url, model_name, node["full_text"], top_k, session
        )
        for child in node.get("children", []):
            _recurse(child)

    _recurse(root)


async def annotate_top_logprobs_async(
    root: Node,
    server_url: str,
    model_name: str,
    top_k: int = 20,
    max_concurrent: int = 8,
    api_key: Optional[str] = None,
) -> None:
    """Store top-K next-token log-probs for every node concurrently.

    Nodes are processed in ascending order of ``full_text`` length (shallowest
    first) so that, when vLLM prefix-caching is enabled
    (``--enable-prefix-caching``), the KV state for each prefix is already
    cached before deeper nodes that share that prefix are processed.
    """
    sem = asyncio.Semaphore(max_concurrent)
    session = _make_session(api_key)

    async def _score(node: Node) -> None:
        async with sem:
            loop = asyncio.get_event_loop()
            node["top_logprobs"] = await loop.run_in_executor(
                None,
                lambda: get_next_token_logprobs(
                    server_url, model_name, node["full_text"], top_k, session
                ),
            )

    all_nodes: List[Node] = []

    def _collect(n: Node) -> None:
        all_nodes.append(n)
        for c in n.get("children", []):
            _collect(c)

    _collect(root)
    # Shallowest nodes first → warms prefix cache for deeper siblings/children
    all_nodes.sort(key=lambda n: len(n.get("full_text", "")))
    await asyncio.gather(*[_score(n) for n in all_nodes])
