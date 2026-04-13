"""
Golden Path Generator -- produce verified reasoning chains from Q+A pairs.

For datasets that only have answers (no step-by-step solutions), this module
generates golden paths by prompting a model with the question + known answer
and asking it to produce reasoning that leads to that answer.

The generated solutions are verified against the ground truth using the
reward functions, and only verified-correct paths are kept.

Supports the same backends as the judge: endpoint (HTTP) or vllm (local).
"""

import asyncio
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


GOLDEN_PATH_SYSTEM = (
    "You are a math tutor. Given a problem and its correct answer, "
    "write a clear step-by-step solution that arrives at the given answer. "
    "Each reasoning step should end with a period. "
    "Put your final answer within \\boxed{}."
)


def _build_golden_prompt(question: str, answer: str) -> List[dict]:
    """Build chat messages for golden path generation."""
    return [
        {"role": "system", "content": GOLDEN_PATH_SYSTEM},
        {"role": "user", "content": (
            f"**Problem:** {question}\n\n"
            f"**Correct Answer:** {answer}\n\n"
            f"Write a detailed step-by-step solution that arrives at this answer."
        )},
    ]


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()[:16]


class GoldenPathGenerator:
    """Generate and cache verified golden paths for answer-only datasets.

    Args:
        endpoint: URL of OpenAI-compatible server (for endpoint mode).
        model: Model name/path.
        mode: "endpoint" or "vllm".
        max_tokens: Max tokens for generation.
        temperature: Sampling temperature (low = more deterministic).
        max_concurrent: Max concurrent requests (endpoint mode).
        max_attempts: Max generation attempts per question before giving up.
        cache_dir: Directory to persist cached golden paths.
    """

    def __init__(
        self,
        endpoint: str = "",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        mode: str = "endpoint",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        max_concurrent: int = 32,
        max_attempts: int = 3,
        cache_dir: str = "data/golden_paths",
    ):
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.model = model
        self.mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_attempts = max_attempts
        self.cache_dir = cache_dir

        # In-memory cache: question_hash -> golden_path_text
        self._cache: Dict[str, str] = {}
        self._load_cache()

        # Lazy-init vLLM
        self._vllm = None
        self._vllm_tokenizer = None
        self._vllm_sampling_params = None

        logger.info(
            f"GoldenPathGenerator: mode={mode}, model={model}, "
            f"max_attempts={max_attempts}"
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, "golden_cache.jsonl")

    def _load_cache(self):
        path = self._cache_path()
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                self._cache[rec["qid"]] = rec["golden_path"]
        logger.info(f"Loaded {len(self._cache)} cached golden paths from {path}")

    def _save_to_cache(self, qid: str, question: str, answer: str,
                       golden_path: str, data_source: str):
        self._cache[qid] = golden_path
        path = self._cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({
                "qid": qid,
                "question": question,
                "answer": answer,
                "golden_path": golden_path,
                "data_source": data_source,
            }, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Generation backends
    # ------------------------------------------------------------------

    def _init_vllm(self):
        if self._vllm is not None:
            return
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self._vllm_tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._vllm = LLM(
            model=self.model,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.2,
        )
        self._vllm_sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
        )
        logger.info(f"Golden path vLLM initialized: {self.model}")

    def _generate_vllm(self, messages_batch: List[List[dict]]) -> List[str]:
        """Generate using local vLLM."""
        self._init_vllm()
        formatted = [
            self._vllm_tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_batch
        ]
        outputs = self._vllm.generate(formatted, self._vllm_sampling_params)
        return [o.outputs[0].text for o in outputs]

    async def _generate_endpoint_async(
        self, messages_batch: List[List[dict]]
    ) -> List[str]:
        """Generate using HTTP endpoint (async)."""
        import aiohttp

        url = f"{self.endpoint}/v1/chat/completions"
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [""] * len(messages_batch)

        async def _request(idx: int, messages: List[dict]):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            for attempt in range(3):
                try:
                    async with semaphore:
                        async with session.post(
                            url, json=payload,
                            timeout=aiohttp.ClientTimeout(total=120),
                        ) as resp:
                            data = await resp.json()
                            results[idx] = data["choices"][0]["message"]["content"]
                            return
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Golden path request failed: {e}")

        async with aiohttp.ClientSession() as session:
            tasks = [_request(i, m) for i, m in enumerate(messages_batch)]
            await asyncio.gather(*tasks)

        return results

    def _generate_endpoint(self, messages_batch: List[List[dict]]) -> List[str]:
        """Sync wrapper for endpoint generation."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._generate_endpoint_async(messages_batch),
                )
                return future.result()
        else:
            return asyncio.run(self._generate_endpoint_async(messages_batch))

    def _generate_batch(self, messages_batch: List[List[dict]]) -> List[str]:
        """Route to the appropriate backend."""
        if self.mode == "vllm":
            return self._generate_vllm(messages_batch)
        elif self.mode == "endpoint":
            return self._generate_endpoint(messages_batch)
        else:
            raise ValueError(f"Unknown golden path mode: {self.mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_golden_paths(
        self,
        questions: List[str],
        answers: List[str],
        data_sources: List[str],
    ) -> List[Optional[str]]:
        """Generate verified golden paths for a batch of Q+A pairs.

        Returns a list of golden path texts (None if generation failed
        after max_attempts or if the question already has a solution).
        """
        # Separate into cached vs needs-generation
        results: List[Optional[str]] = [None] * len(questions)
        to_generate: List[Tuple[int, str, str, str, str]] = []  # (idx, qid, q, a, ds)

        for i, (q, a, ds) in enumerate(zip(questions, answers, data_sources)):
            if not a.strip():
                # No answer available -- can't generate golden path
                continue
            qid = _question_hash(q)
            if qid in self._cache:
                results[i] = self._cache[qid]
            else:
                to_generate.append((i, qid, q, a, ds))

        if not to_generate:
            n_cached = sum(1 for r in results if r is not None)
            if n_cached > 0:
                logger.info(f"Golden paths: {n_cached} from cache, 0 to generate")
            return results

        logger.info(
            f"Golden paths: {len(results) - len(to_generate)} cached, "
            f"{len(to_generate)} to generate"
        )

        # Generate with retries
        remaining = list(to_generate)

        for attempt in range(self.max_attempts):
            if not remaining:
                break

            messages_batch = [
                _build_golden_prompt(q, a) for (_, _, q, a, _) in remaining
            ]
            generated = self._generate_batch(messages_batch)

            still_remaining = []
            for (idx, qid, q, a, ds), text in zip(remaining, generated):
                if not text.strip():
                    still_remaining.append((idx, qid, q, a, ds))
                    continue

                # Verify: does the generated solution produce the correct answer?
                score = compute_score(
                    data_source=ds, solution_str=text, ground_truth=a
                )
                if score >= 1.0:
                    results[idx] = text
                    self._save_to_cache(qid, q, a, text, ds)
                else:
                    still_remaining.append((idx, qid, q, a, ds))

            n_success = len(remaining) - len(still_remaining)
            logger.info(
                f"Golden path attempt {attempt + 1}/{self.max_attempts}: "
                f"{n_success}/{len(remaining)} verified correct"
            )
            remaining = still_remaining

        if remaining:
            logger.warning(
                f"Failed to generate verified golden paths for "
                f"{len(remaining)} questions after {self.max_attempts} attempts"
            )

        return results
