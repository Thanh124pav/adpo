"""
LLM-as-Judge for per-phase reward scoring.

Given a math question and a response segmented into phases, the judge
evaluates each phase independently and returns a reward vector.

Supports:
- vLLM-based local judge (fast, batched)
- API-based judge (OpenAI-compatible endpoint)
- Rule-based fallback (for final-answer phases)
"""

import re
import json
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from adpo.adpo_algorithm import PhaseSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge Prompts
# ---------------------------------------------------------------------------

PHASE_JUDGE_SYSTEM = (
    "You are a math reasoning evaluator. You will be given a math problem, "
    "the reasoning context so far, and a specific reasoning step to evaluate. "
    "Score the reasoning step on correctness and usefulness.\n\n"
    "Output ONLY a JSON object: {\"score\": <float between 0.0 and 1.0>, "
    "\"reason\": \"<brief explanation>\"}\n\n"
    "Scoring guide:\n"
    "- 1.0: Step is correct, logical, and advances toward the solution\n"
    "- 0.7-0.9: Step is mostly correct with minor issues\n"
    "- 0.4-0.6: Step has some correct ideas but also errors\n"
    "- 0.1-0.3: Step is mostly incorrect or irrelevant\n"
    "- 0.0: Step is completely wrong or contradicts previous correct work"
)


def build_phase_judge_prompt(
    question: str,
    context_phases: List[str],
    current_phase: str,
    phase_idx: int,
    total_phases: int,
) -> List[dict]:
    """Build the judge prompt for evaluating a single phase.

    Args:
        question: The original math problem.
        context_phases: Text of all phases before the current one.
        current_phase: Text of the phase being evaluated.
        phase_idx: 0-indexed phase number.
        total_phases: Total number of phases.

    Returns:
        Chat messages for the judge.
    """
    context_text = ""
    if context_phases:
        context_text = (
            "**Previous reasoning steps:**\n"
            + "\n---\n".join(
                f"Step {i+1}: {p}" for i, p in enumerate(context_phases)
            )
            + "\n\n"
        )

    user_msg = (
        f"**Math Problem:**\n{question}\n\n"
        f"{context_text}"
        f"**Step {phase_idx + 1} of {total_phases} to evaluate:**\n"
        f"{current_phase}\n\n"
        f"Evaluate this reasoning step. Output JSON: "
        f'{{\"score\": <0.0-1.0>, \"reason\": \"...\"}}'
    )

    return [
        {"role": "system", "content": PHASE_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_judge_response(response_text: str) -> float:
    """Extract score from judge response JSON."""
    # Try direct JSON parse
    try:
        data = json.loads(response_text.strip())
        return float(data["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try to find JSON in response
    json_match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', response_text)
    if json_match:
        try:
            return float(json_match.group(1))
        except ValueError:
            pass

    # Try to find any float
    float_match = re.search(r'(\d+\.?\d*)', response_text)
    if float_match:
        val = float(float_match.group(1))
        if 0.0 <= val <= 1.0:
            return val

    logger.warning(f"Could not parse judge response: {response_text[:100]}")
    return 0.5  # Default neutral score


# ---------------------------------------------------------------------------
# Judge Backends
# ---------------------------------------------------------------------------

class PhaseJudge:
    """Base class for phase judges."""

    def score_phases(
        self,
        questions: List[str],
        phase_texts: List[List[str]],
    ) -> List[List[float]]:
        """Score phases for a batch of responses.

        Args:
            questions: List of math problems, one per response.
            phase_texts: List of phase text lists.
                phase_texts[i] = ["phase 0 text", "phase 1 text", ...]

        Returns:
            List of reward lists.
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
        logger.info(f"VLLMPhaseJudge initialized with {model_path}")

    def score_phases(self, questions, phase_texts):
        # Build all prompts
        all_prompts = []
        prompt_map = []  # (response_idx, phase_idx)

        for i, (question, phases) in enumerate(zip(questions, phase_texts)):
            for k, phase_text in enumerate(phases):
                context = phases[:k]
                messages = build_phase_judge_prompt(
                    question=question,
                    context_phases=context,
                    current_phase=phase_text,
                    phase_idx=k,
                    total_phases=len(phases),
                )
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                all_prompts.append(formatted)
                prompt_map.append((i, k))

        if not all_prompts:
            return [[] for _ in questions]

        # Batch inference
        outputs = self.llm.generate(all_prompts, self.sampling_params)

        # Parse results
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
    ):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        logger.info(f"APIPhaseJudge initialized with {model}")

    def _score_single(self, question, phases, k):
        context = phases[:k]
        messages = build_phase_judge_prompt(
            question=question,
            context_phases=context,
            current_phase=phases[k],
            phase_idx=k,
            total_phases=len(phases),
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

    def score_phases(self, questions, phase_texts):
        import concurrent.futures

        rewards = [[] for _ in questions]
        tasks = []

        for i, (question, phases) in enumerate(zip(questions, phase_texts)):
            for k in range(len(phases)):
                tasks.append((i, k, question, phases))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent
        ) as executor:
            futures = {
                executor.submit(self._score_single, t[2], t[3], t[1]): (t[0], t[1])
                for t in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                i, k = futures[future]
                try:
                    score = future.result()
                except Exception:
                    score = 0.5
                rewards[i].append(score)

        # Sort rewards by phase index (futures may complete out of order)
        for i in range(len(rewards)):
            # Rewards were appended in completion order; re-sort needed
            # Actually we need a different approach - let me fix:
            pass

        # Simpler: sequential for correctness
        rewards = [[] for _ in questions]
        for i, (question, phases) in enumerate(zip(questions, phase_texts)):
            for k in range(len(phases)):
                score = self._score_single(question, phases, k)
                rewards[i].append(score)

        return rewards


class RuleBasedPhaseJudge(PhaseJudge):
    """Fallback: only score the final phase using answer matching.

    Intermediate phases get a neutral score of 0.5.
    The final phase gets 1.0 if the answer is correct, 0.0 otherwise.
    This is a baseline that approximates standard GRPO behavior.
    """

    def __init__(self, reward_fn=None):
        from adpo.reward_functions import compute_score
        self.reward_fn = reward_fn or compute_score

    def score_phases(self, questions, phase_texts, ground_truths=None,
                     data_sources=None):
        rewards = []
        for i, phases in enumerate(phase_texts):
            n = len(phases)
            phase_rewards = [0.5] * n  # neutral for intermediate

            if ground_truths is not None and i < len(ground_truths):
                # Score final phase using answer matching
                full_response = " ".join(phases)
                final_score = self.reward_fn(
                    data_source=data_sources[i] if data_sources else "math",
                    solution_str=full_response,
                    ground_truth=ground_truths[i],
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

    Args:
        judge_type: "vllm", "api", or "rule".
        judge_model: Model path or name.
    """
    if judge_type == "vllm":
        return VLLMPhaseJudge(model_path=judge_model, **kwargs)
    elif judge_type == "api":
        return APIPhaseJudge(model=judge_model, **kwargs)
    elif judge_type == "rule":
        return RuleBasedPhaseJudge(**kwargs)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")
