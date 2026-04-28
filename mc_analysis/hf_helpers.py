"""
HuggingFace backend — equivalent of vllm_helpers but runs locally via
transformers AutoModelForCausalLM.

Public API mirrors vllm_helpers:
    HFBackend.rollout_with_alpha(prompt, K, ...)
    HFBackend.compute_sequence_logprob(prompt, completion)
    HFBackend.build_tree(prompt, ...)
    HFBackend.build_tree_async(prompt, ...)   # runs sync DFS inside asyncio executor

Standalone helpers (require model + tokenizer passed explicitly):
    rollout_with_alpha_hf(model, tokenizer, prompt, K, ...)
    compute_sequence_logprob_hf(model, tokenizer, prompt, completion)
    build_tree_hf(model, tokenizer, prompt, ...)
"""

import asyncio
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

Node = Dict[str, Any]


# =============================================================================
# Internal helpers
# =============================================================================

def _log_softmax(logits):
    """numpy log_softmax along last axis."""
    m = logits.max(axis=-1, keepdims=True)
    log_p = logits - m - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))
    return log_p


def _resolve_branch_factor(branch_factor: Union[int, List[int]], depth: int) -> int:
    if isinstance(branch_factor, int):
        return branch_factor
    return branch_factor[min(depth, len(branch_factor) - 1)]


def _is_terminal(node: Node, stop: Optional[List[str]]) -> bool:
    if node["finish_reason"] == "length":
        return True
    if stop and any(node["text"].endswith(s) for s in stop):
        return False
    return True


def _assign_answer(node: Node, extract_fn: Optional[Callable]) -> None:
    if extract_fn is not None:
        ans = extract_fn(node["text"])
        if ans is not None:
            node["answer"] = ans
    else:
        node["answer"] = node["text"]


# =============================================================================
# Core functions (require model + tokenizer as arguments)
# =============================================================================

def rollout_with_alpha_hf(
    model,
    tokenizer,
    prompt: str,
    K: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate one sequence with HuggingFace and annotate every token position i:

        α_i = mean(logprob[i], …, logprob[min(i+K-1, n-1)])
        percentile_i = #{j ≤ i : α_j ≤ α_i} / (i+1) × 100

    Parameters
    ----------
    model      : AutoModelForCausalLM (already on device, eval mode)
    tokenizer  : AutoTokenizer
    prompt     : str
    K          : sliding-window width
    max_new_tokens, temperature, top_p, seed : generation kwargs

    Returns
    -------
    dict with keys: text, tokens, logprobs, alpha, percentile, finish_reason
    """
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prefix_len = enc.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    gen_ids = out.sequences[0][prefix_len:]   # (n_new_tokens,)
    scores   = out.scores                      # tuple of (1, vocab) tensors

    tokens: List[str] = []
    logprobs: List[float] = []

    for step, token_id in enumerate(gen_ids.tolist()):
        if step >= len(scores):
            break
        lp = float(F.log_softmax(scores[step][0], dim=-1)[token_id])
        logprobs.append(lp)
        tokens.append(tokenizer.decode([token_id]))

    n = len(logprobs)
    alphas = [float(np.mean(logprobs[i: i + K])) for i in range(n)]
    alphas_arr = np.asarray(alphas)
    percentiles = [
        float(np.sum(alphas_arr[: i + 1] <= alphas_arr[i])) / (i + 1) * 100.0
        for i in range(n)
    ]

    eos_id = tokenizer.eos_token_id
    finish_reason = "stop" if (len(gen_ids) > 0 and gen_ids[-1].item() == eos_id) else "length"

    return {
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "tokens": tokens,
        "logprobs": logprobs,
        "alpha": alphas,
        "percentile": percentiles,
        "finish_reason": finish_reason,
    }


def compute_sequence_logprob_hf(
    model,
    tokenizer,
    prompt: str,
    completion: str,
) -> Dict[str, Any]:
    """
    Compute log P(completion | prompt) with a HuggingFace model.

    Tokenises (prompt + completion) jointly, does one forward pass, and
    extracts the log-probs of the completion tokens conditioned on the prompt.

    Returns
    -------
    dict: tokens, logprobs, sum_logprob, mean_logprob, num_tokens
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device

    prompt_ids     = tokenizer(prompt,     add_special_tokens=True ).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids

    full_ids = prompt_ids + completion_ids
    input_t  = torch.tensor([full_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_t).logits[0]           # (seq_len, vocab)

    log_probs = F.log_softmax(logits, dim=-1)       # (seq_len, vocab)

    n_prompt     = len(prompt_ids)
    n_completion = len(completion_ids)

    comp_lps: List[float] = []
    for i in range(n_completion):
        # logits at position (n_prompt + i - 1) predict token at (n_prompt + i)
        pos = n_prompt + i - 1
        tok = completion_ids[i]
        comp_lps.append(float(log_probs[pos, tok]))

    comp_tokens = [tokenizer.decode([t]) for t in completion_ids]
    sum_lp  = float(sum(comp_lps))
    mean_lp = sum_lp / len(comp_lps) if comp_lps else 0.0

    return {
        "tokens": comp_tokens,
        "logprobs": comp_lps,
        "sum_logprob": sum_lp,
        "mean_logprob": mean_lp,
        "num_tokens": len(comp_lps),
    }


def _sample_completions_hf(
    model,
    tokenizer,
    prefix: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    get_logprobs: bool,
    seed: Optional[int],
) -> List[Node]:
    """Sample n completions using HF model.generate() with num_return_sequences."""
    import torch
    import torch.nn.functional as F
    from transformers import StoppingCriteria, StoppingCriteriaList

    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    enc    = tokenizer(prefix, return_tensors="pt").to(device)
    prefix_len = enc.input_ids.shape[1]

    # Stop-string criteria
    stopping = None
    if stop:
        class _StopOnStrings(StoppingCriteria):
            def __init__(self, tok, strings, p_len):
                self.tok = tok
                self.strings = strings
                self.p_len = p_len
            def __call__(self, input_ids, scores, **kw):
                for seq in input_ids:
                    gen_text = self.tok.decode(seq[self.p_len:], skip_special_tokens=False)
                    if any(s in gen_text for s in self.strings):
                        return True
                return False
        stopping = StoppingCriteriaList([_StopOnStrings(tokenizer, stop, prefix_len)])

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        do_sample=(temperature > 0 or n > 1),
        temperature=max(temperature, 1e-4),
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=get_logprobs,
        pad_token_id=tokenizer.eos_token_id,
    )
    if stopping is not None:
        gen_kwargs["stopping_criteria"] = stopping

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    eos_id = tokenizer.eos_token_id
    nodes: List[Node] = []

    for seq_idx in range(n):
        gen_ids = out.sequences[seq_idx][prefix_len:]
        text    = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Determine finish reason
        last_tok = gen_ids[-1].item() if len(gen_ids) > 0 else -1
        if last_tok == eos_id:
            finish_reason = "stop"
        elif stop and any(text.endswith(s) for s in stop):
            finish_reason = "stop"
        else:
            finish_reason = "length"

        node: Node = {
            "text": text,
            "full_text": prefix + text,
            "finish_reason": finish_reason,
        }

        if get_logprobs and out.scores:
            lps: List[float] = []
            for step, tok_id in enumerate(gen_ids.tolist()):
                if step >= len(out.scores):
                    break
                # scores[step] shape: (num_return_sequences, vocab)
                lp = float(F.log_softmax(out.scores[step][seq_idx], dim=-1)[tok_id])
                lps.append(lp)
            node["sum_logprobs"] = float(sum(lps))
            node["num_tokens"]   = len(lps)
            node["token_logprobs"] = lps
            node["tokens"] = [tokenizer.decode([t]) for t in gen_ids.tolist()[:len(lps)]]

        nodes.append(node)

    return nodes


def build_tree_hf(
    model,
    tokenizer,
    prompt: str,
    max_depth: int = 2,
    branch_factor: Union[int, List[int]] = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    get_logprobs: bool = True,
    seed: Optional[int] = None,
) -> Node:
    """
    Build a search tree using HuggingFace model.generate().

    Same Node structure and semantics as vllm_helpers.build_tree.
    Branching uses num_return_sequences; stop sequences are enforced via
    StoppingCriteria (all sequences stop when any hits a stop string).
    """
    def _dfs(node: Node, depth: int) -> None:
        if depth >= max_depth:
            _assign_answer(node, extract_answer_fn)
            return

        bf = _resolve_branch_factor(branch_factor, depth)
        children = _sample_completions_hf(
            model, tokenizer,
            prefix=node["full_text"],
            n=bf,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            get_logprobs=get_logprobs,
            seed=seed,
        )
        for child in children:
            child["depth"] = depth + 1
            if _is_terminal(child, stop):
                _assign_answer(child, extract_answer_fn)
            else:
                _dfs(child, depth + 1)
        node["children"] = children

    root: Node = {"text": prompt, "full_text": prompt, "depth": 0}
    _dfs(root, 0)
    return root


async def build_tree_hf_async(
    model,
    tokenizer,
    prompt: str,
    max_depth: int = 2,
    branch_factor: Union[int, List[int]] = 3,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
    get_logprobs: bool = True,
    seed: Optional[int] = None,
    max_concurrent: int = 4,
) -> Node:
    """
    Async variant of build_tree_hf — runs HF inference in thread-pool
    executors so the event loop stays responsive.

    Note: GPU inference is not truly parallel (GIL + CUDA serialisation),
    but this allows interleaving with other async work.
    max_concurrent limits simultaneous generate() calls.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def _async_sample(prefix: str, n: int) -> List[Node]:
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: _sample_completions_hf(
                    model, tokenizer, prefix, n,
                    max_new_tokens, temperature, top_p,
                    stop, get_logprobs, seed,
                ),
            )

    async def _dfs(node: Node, depth: int) -> None:
        if depth >= max_depth:
            _assign_answer(node, extract_answer_fn)
            return
        bf = _resolve_branch_factor(branch_factor, depth)
        children = await _async_sample(node["full_text"], bf)
        tasks = []
        for child in children:
            child["depth"] = depth + 1
            if _is_terminal(child, stop):
                _assign_answer(child, extract_answer_fn)
            else:
                tasks.append(asyncio.create_task(_dfs(child, depth + 1)))
        if tasks:
            await asyncio.gather(*tasks)
        node["children"] = children

    root: Node = {"text": prompt, "full_text": prompt, "depth": 0}
    await _dfs(root, 0)
    return root


# =============================================================================
# Top-K next-token log-probs  (used for JSD between siblings)
# =============================================================================

def get_next_token_logprobs_hf(
    model,
    tokenizer,
    text: str,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Get top-K next-token log-probs at the end of *text* using a HF model.

    Returns
    -------
    List[Tuple[str, float]]
        ``[(token_string, log_prob), ...]`` sorted by descending log-prob.
    """
    import torch

    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    log_probs = torch.log_softmax(out.logits[0, -1, :], dim=-1)
    k = min(top_k, log_probs.shape[-1])
    top_lps, top_ids = torch.topk(log_probs, k)
    return [
        (tokenizer.decode([int(tid)]), float(lp))
        for tid, lp in zip(top_ids.tolist(), top_lps.tolist())
    ]


def annotate_top_logprobs_hf(
    model,
    tokenizer,
    root: Node,
    top_k: int = 20,
) -> None:
    """Store top-K next-token log-probs in ``node["top_logprobs"]`` for every node (sequential DFS)."""
    def _recurse(node: Node) -> None:
        node["top_logprobs"] = get_next_token_logprobs_hf(
            model, tokenizer, node["full_text"], top_k
        )
        for child in node.get("children", []):
            _recurse(child)

    _recurse(root)


# =============================================================================
# HFBackend class — convenient wrapper that holds model + tokenizer
# =============================================================================

class HFBackend:
    """
    Wrapper around a HuggingFace causal LM that exposes the same interface
    as the vLLM REST backend.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id or local path.
    device : str
        "cuda", "cuda:0", "cpu", etc.
    dtype : torch.dtype, optional
        Model weight dtype (default: float16 on CUDA, float32 on CPU).
    load_in_8bit / load_in_4bit : bool
        bitsandbytes quantisation flags (require bitsandbytes installed).
    kwargs
        Extra args forwarded to AutoModelForCausalLM.from_pretrained.

    Usage
    -----
        backend = HFBackend("Qwen/Qwen2.5-7B-Instruct", device="cuda")
        result  = backend.rollout_with_alpha("What is 2+2?", K=5)
        tree    = backend.build_tree("What is 2+2?", max_depth=2, branch_factor=3)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype=None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if dtype is None:
            dtype = torch.float16 if "cuda" in device else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kw: Dict[str, Any] = dict(
            trust_remote_code=True,
            **kwargs,
        )

        # transformers ≥ 4.30 (including 5.x): quantization via BitsAndBytesConfig
        if load_in_4bit:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
            load_kw["device_map"] = device
        elif load_in_8bit:
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kw["device_map"] = device
        else:
            load_kw["torch_dtype"] = dtype
            load_kw["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kw
        )
        self.model.eval()
        self.device = device

    # ── public methods ────────────────────────────────────────────────────────

    def rollout_with_alpha(self, prompt: str, K: int = 10, **kwargs) -> Dict[str, Any]:
        """See :func:`rollout_with_alpha_hf`."""
        return rollout_with_alpha_hf(self.model, self.tokenizer, prompt, K=K, **kwargs)

    def compute_sequence_logprob(self, prompt: str, completion: str) -> Dict[str, Any]:
        """See :func:`compute_sequence_logprob_hf`."""
        return compute_sequence_logprob_hf(self.model, self.tokenizer, prompt, completion)

    def build_tree(self, prompt: str, **kwargs) -> Node:
        """See :func:`build_tree_hf`."""
        return build_tree_hf(self.model, self.tokenizer, prompt, **kwargs)

    async def build_tree_async(self, prompt: str, **kwargs) -> Node:
        """See :func:`build_tree_hf_async`."""
        return await build_tree_hf_async(self.model, self.tokenizer, prompt, **kwargs)

    def compute_p_for_node(self, node: Node, gold_answer: str) -> float:
        """
        P(node) = log P(gold_answer | trajectory_to_node)
        Stores result in node["P"] and returns it.
        """
        result = self.compute_sequence_logprob(
            prompt=node["full_text"],
            completion=gold_answer,
        )
        node["P"] = float(result["sum_logprob"])
        return node["P"]

    def compute_p_tree(self, root: Node, gold_answer: str) -> None:
        """Compute P for every node in the tree (sequential)."""
        self.compute_p_for_node(root, gold_answer)
        for child in root.get("children", []):
            self.compute_p_tree(child, gold_answer)

    async def compute_p_tree_async(
        self,
        root: Node,
        gold_answer: str,
        max_concurrent: int = 4,
    ) -> None:
        """Compute P for every node concurrently (async, thread-pool)."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _score(node: Node) -> None:
            async with sem:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.compute_p_for_node(node, gold_answer),
                )

        all_nodes: List[Node] = []

        def _collect(n: Node) -> None:
            all_nodes.append(n)
            for c in n.get("children", []):
                _collect(c)

        _collect(root)
        await asyncio.gather(*[_score(n) for n in all_nodes])

    def get_next_token_logprobs(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """See :func:`get_next_token_logprobs_hf`."""
        return get_next_token_logprobs_hf(self.model, self.tokenizer, text, top_k=top_k)

    def annotate_top_logprobs(self, root: Node, top_k: int = 20) -> None:
        """See :func:`annotate_top_logprobs_hf`."""
        annotate_top_logprobs_hf(self.model, self.tokenizer, root, top_k=top_k)

    async def annotate_top_logprobs_async(
        self,
        root: Node,
        top_k: int = 20,
        max_concurrent: int = 4,
    ) -> None:
        """Annotate top-K next-token log-probs concurrently (thread-pool)."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _score(node: Node) -> None:
            async with sem:
                loop = asyncio.get_event_loop()
                node["top_logprobs"] = await loop.run_in_executor(
                    None,
                    lambda: get_next_token_logprobs_hf(
                        self.model, self.tokenizer, node["full_text"], top_k
                    ),
                )

        all_nodes: List[Node] = []

        def _collect(n: Node) -> None:
            all_nodes.append(n)
            for c in n.get("children", []):
                _collect(c)

        _collect(root)
        await asyncio.gather(*[_score(n) for n in all_nodes])
