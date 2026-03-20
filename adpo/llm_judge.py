"""
LLM-as-Judge for per-phase reward scoring.

Given a math question and a response segmented into phases, the judge
evaluates each phase independently and returns a reward vector.

Judge context includes:
- The original math problem
- The correct answer (golden answer) if available
- Reference solutions from the SolutionBank (correct solutions generated
  by teacher model or discovered during training)
- Previous reasoning phases (causal context)

Supports:
- vLLM-based local judge (fast, batched)
- API-based judge (OpenAI-compatible endpoint)
- Rule-based fallback (for final-answer phases)
"""

import re
import json
import asyncio
import logging
from typing import List, Optional

from adpo.adpo_algorithm import PhaseSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge Prompts
# ---------------------------------------------------------------------------

PHASE_JUDGE_SYSTEM = (
    "You are a math reasoning evaluator. You will be given:\n"
    "1. A math problem\n"
    "2. The correct answer (if available)\n"
    "3. Reference solutions showing correct approaches (if available)\n"
    "4. A specific reasoning step to evaluate\n\n"
    "Your job: evaluate whether the reasoning step is correct and "
    "moves toward the correct answer.\n\n"
    "Output ONLY a JSON object: {\"score\": <float between 0.0 and 1.0>, "
    "\"reason\": \"<brief explanation>\"}\n\n"
    "Scoring guide:\n"
    "- 1.0: Step is mathematically correct and clearly advances toward the answer\n"
    "- 0.7-0.9: Step is correct but could be more efficient or clear\n"
    "- 0.4-0.6: Step is correct but there is an other step that advances toward the answer more efficient"
    "- 0.1-0.3: Step has partial correctness but contains errors or goes off track\n"
    "- 0.0: Step is completely wrong or leads away from the correct answer"
)


def build_phase_judge_prompt(
    question: str,
    context_phases: List[str],
    current_phase: str,
    phase_idx: int,
    total_phases: int,
    golden_answer: str = "",
    reference_solutions: Optional[List[str]] = None,
) -> List[dict]:
    """Build the judge prompt for evaluating a single phase.

    Args:
        question: The original math problem.
        context_phases: Text of all phases before the current one.
        current_phase: Text of the phase being evaluated.
        phase_idx: 0-indexed phase number.
        total_phases: Total number of phases.
        golden_answer: The correct answer (if known).
        reference_solutions: List of correct solution texts (from SolutionBank).

    Returns:
        Chat messages for the judge.
    """
    parts = [f"**Math Problem:**\n{question}\n"]

    # Golden answer
    if golden_answer:
        parts.append(f"**Correct Answer:** {golden_answer}\n")

    # Reference solutions
    if reference_solutions:
        n_refs = len(reference_solutions)
        parts.append(f"**Reference Solutions ({n_refs} correct approach(es)):**")
        for i, sol in enumerate(reference_solutions):
            # Truncate long solutions to keep prompt manageable
            truncated = sol[:1500] + "..." if len(sol) > 1500 else sol
            parts.append(f"\n--- Solution {i+1} ---\n{truncated}")
        parts.append("")

    # Previous context
    if context_phases:
        parts.append("**Previous reasoning steps (from the response being evaluated):**")
        for i, p in enumerate(context_phases):
            parts.append(f"Step {i+1}: {p}")
        parts.append("")

    # Current phase to evaluate
    parts.append(
        f"**Step {phase_idx + 1} of {total_phases} to evaluate:**\n"
        f"{current_phase}\n\n"
        f"Evaluate this reasoning step. Consider whether it aligns with "
        f"the correct answer and reference solutions above. "
        f"Output JSON: {{\"score\": <0.0-1.0>, \"reason\": \"...\"}}"
    )

    user_msg = "\n".join(parts)

    return [
        {"role": "system", "content": PHASE_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_judge_response(response_text: str) -> float:
    """Extract score from judge response JSON."""
    # Strip thinking tags (e.g. <think>...</think>) from reasoning models
    text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    if not text:
        text = response_text  # fallback if everything was inside think tags

    try:
        data = json.loads(text.strip())
        return float(data["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    json_match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', text)
    if json_match:
        try:
            return float(json_match.group(1))
        except ValueError:
            pass

    float_match = re.search(r'(\d+\.?\d*)', text)
    if float_match:
        val = float(float_match.group(1))
        if 0.0 <= val <= 1.0:
            return val

    logger.warning(f"Could not parse judge response: {response_text[:100]}")
    return 0.5


# ---------------------------------------------------------------------------
# Judge Backends
# ---------------------------------------------------------------------------

class PhaseJudge:
    """Base class for phase judges."""

    def score_phases(
        self,
        questions: List[str],
        phase_texts: List[List[str]],
        golden_answers: Optional[List[str]] = None,
        reference_solutions: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Score phases for a batch of responses.

        Args:
            questions: List of math problems, one per response.
            phase_texts: phase_texts[i] = ["phase 0 text", "phase 1 text", ...]
            golden_answers: Golden answers per response (if available).
            reference_solutions: Reference solutions per response from SolutionBank.
                reference_solutions[i] = ["correct sol 1", "correct sol 2", ...]

        Returns:
            rewards[i] = [r_0, r_1, ...] for response i.
        """
        raise NotImplementedError


class VLLMPhaseJudge(PhaseJudge):
    """Judge using a local vLLM model for fast batched inference."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_ref_solutions_in_prompt: int = 3,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.3,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.max_ref_solutions_in_prompt = max_ref_solutions_in_prompt
        logger.info(f"VLLMPhaseJudge initialized with {model_path}")

    def score_phases(self, questions, phase_texts, golden_answers=None,
                     reference_solutions=None, **kwargs):
        all_prompts = []
        prompt_map = []

        for i, (question, phases) in enumerate(zip(questions, phase_texts)):
            ga = golden_answers[i] if golden_answers and i < len(golden_answers) else ""
            refs = None
            if reference_solutions and i < len(reference_solutions):
                refs = reference_solutions[i][:self.max_ref_solutions_in_prompt]

            for k, phase_text in enumerate(phases):
                context = phases[:k]
                messages = build_phase_judge_prompt(
                    question=question,
                    context_phases=context,
                    current_phase=phase_text,
                    phase_idx=k,
                    total_phases=len(phases),
                    golden_answer=ga,
                    reference_solutions=refs,
                )
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                all_prompts.append(formatted)
                prompt_map.append((i, k))

        if not all_prompts:
            return [[] for _ in questions]

        outputs = self.llm.generate(all_prompts, self.sampling_params)

        rewards = [[] for _ in questions]
        for output, (i, k) in zip(outputs, prompt_map):
            response_text = output.outputs[0].text
            score = parse_judge_response(response_text)
            rewards[i].append(score)

        return rewards


class APIPhaseJudge(PhaseJudge):
    """Judge using an OpenAI-compatible API endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_concurrent: int = 32,
        max_ref_solutions_in_prompt: int = 3,
    ):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_ref_solutions_in_prompt = max_ref_solutions_in_prompt
        logger.info(f"APIPhaseJudge initialized with {model}")

    def _score_single(self, question, phases, k, golden_answer="", refs=None):
        context = phases[:k]
        messages = build_phase_judge_prompt(
            question=question,
            context_phases=context,
            current_phase=phases[k],
            phase_idx=k,
            total_phases=len(phases),
            golden_answer=golden_answer,
            reference_solutions=refs,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return parse_judge_response(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"API judge error: {e}")
            return 0.5

    def score_phases(self, questions, phase_texts, golden_answers=None,
                     reference_solutions=None, **kwargs):
        import concurrent.futures

        tasks = []
        for i, (question, phases) in enumerate(zip(questions, phase_texts)):
            ga = golden_answers[i] if golden_answers and i < len(golden_answers) else ""
            refs = None
            if reference_solutions and i < len(reference_solutions):
                refs = reference_solutions[i][:self.max_ref_solutions_in_prompt]

            for k in range(len(phases)):
                tasks.append((i, k, question, phases, ga, refs))

        rewards = [[None] * len(phase_texts[i]) for i in range(len(questions))]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent
        ) as executor:
            futures = {}
            for task in tasks:
                i, k, question, phases, ga, refs = task
                future = executor.submit(
                    self._score_single, question, phases, k, ga, refs
                )
                futures[future] = (i, k)

            for future in concurrent.futures.as_completed(futures):
                i, k = futures[future]
                try:
                    score = future.result()
                except Exception:
                    score = 0.5
                rewards[i][k] = score

        # Fill any None values
        for i in range(len(rewards)):
            for k in range(len(rewards[i])):
                if rewards[i][k] is None:
                    rewards[i][k] = 0.5

        return rewards


class EndpointPhaseJudge(PhaseJudge):
    """Judge using a raw HTTP endpoint (aiohttp), no openai dependency.

    Compatible with any OpenAI-compatible server (vLLM, TGI, llama.cpp, etc.).
    Uses async HTTP with concurrency control, similar to generate_solutions_api.py.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_concurrent: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        max_ref_solutions_in_prompt: int = 3,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.completions_url = f"{self.endpoint}/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.max_ref_solutions_in_prompt = max_ref_solutions_in_prompt
        logger.info(
            f"EndpointPhaseJudge initialized: model={model}, "
            f"endpoint={self.endpoint}, max_concurrent={max_concurrent}"
        )

    async def _request_single(self, session, semaphore, messages):
        """Send a single chat completion request with retries."""
        import aiohttp as _aiohttp

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    async with session.post(
                        self.completions_url,
                        json=payload,
                        timeout=_aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            text = data["choices"][0]["message"]["content"]
                            return parse_judge_response(text)
                        elif resp.status == 429:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        else:
                            body = await resp.text()
                            logger.warning(f"Judge endpoint error {resp.status}: {body[:200]}")
                            await asyncio.sleep(self.retry_delay)
            except (Exception,) as e:
                logger.warning(f"Judge request error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        logger.warning("Judge request failed after all retries, returning 0.5")
        return 0.5

    async def _score_batch_async(self, all_messages):
        """Score all messages concurrently."""
        import aiohttp as _aiohttp

        semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = _aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
        )
        async with _aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._request_single(session, semaphore, msgs)
                for msgs in all_messages
            ]
            return await asyncio.gather(*tasks)

    def score_phases(self, questions, phase_texts, golden_answers=None,
                     reference_solutions=None, **kwargs):
        import asyncio

        all_messages = []
        prompt_map = []

        for i, (question, phases) in enumerate(zip(questions, phase_texts)):
            ga = golden_answers[i] if golden_answers and i < len(golden_answers) else ""
            refs = None
            if reference_solutions and i < len(reference_solutions):
                refs = reference_solutions[i][:self.max_ref_solutions_in_prompt]

            for k in range(len(phases)):
                messages = build_phase_judge_prompt(
                    question=question,
                    context_phases=phases[:k],
                    current_phase=phases[k],
                    phase_idx=k,
                    total_phases=len(phases),
                    golden_answer=ga,
                    reference_solutions=refs,
                )
                all_messages.append(messages)
                prompt_map.append((i, k))

        if not all_messages:
            return [[] for _ in questions]

        # Run async event loop (handle case where loop is already running)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                scores = pool.submit(
                    asyncio.run, self._score_batch_async(all_messages)
                ).result()
        else:
            scores = asyncio.run(self._score_batch_async(all_messages))

        # Map scores back to [response][phase] structure
        rewards = [[0.5] * len(phase_texts[i]) for i in range(len(questions))]
        for score, (i, k) in zip(scores, prompt_map):
            rewards[i][k] = score

        return rewards


class RuleBasedPhaseJudge(PhaseJudge):
    """Fallback: score final phase using answer matching.

    Intermediate phases get 0.5 (neutral).
    Final phase gets 1.0 if answer correct, 0.0 otherwise.
    """

    def __init__(self, reward_fn=None):
        from adpo.reward_functions import compute_score
        self.reward_fn = reward_fn or compute_score

    def score_phases(self, questions, phase_texts, golden_answers=None,
                     reference_solutions=None, data_sources=None, **kwargs):
        rewards = []
        for i, phases in enumerate(phase_texts):
            n = len(phases)
            phase_rewards = [0.5] * n

            if golden_answers is not None and i < len(golden_answers):
                full_response = " ".join(phases)
                final_score = self.reward_fn(
                    data_source=data_sources[i] if data_sources else "math",
                    solution_str=full_response,
                    ground_truth=golden_answers[i],
                )
                phase_rewards[-1] = final_score

            rewards.append(phase_rewards)
        return rewards


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_judge(
    judge_type: str = "rule",
    judge_model: str = "Qwen/Qwen2.5-7B-Instruct",
    **kwargs,
) -> PhaseJudge:
    """Create a phase judge by type.

    Types:
        rule     - Answer-matching only (no LLM, fast).
        vllm     - Local vLLM model for batched inference.
        api      - OpenAI SDK client (requires `openai` package).
        endpoint - Raw HTTP to any OpenAI-compatible server (uses aiohttp).
    """
    if judge_type == "vllm":
        return VLLMPhaseJudge(model_path=judge_model, **kwargs)
    elif judge_type == "api":
        return APIPhaseJudge(model=judge_model, **kwargs)
    elif judge_type == "endpoint":
        return EndpointPhaseJudge(model=judge_model, **kwargs)
    elif judge_type == "rule":
        return RuleBasedPhaseJudge(**kwargs)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")
