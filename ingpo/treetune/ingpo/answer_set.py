"""Build the per-problem answer set Y at depth=1.

PLAN Def 2.1: Y = LLM(p, "Given answer=a*, list m diverse step-by-step
solutions"). Y is reused for the entire tree of one problem.

We piggy-back on the SPO/guidance vLLM client by issuing a single sampling
call with `n=m` and temperature=0.7.  The prompt template can be customised
per task family.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional


DEFAULT_Y_PROMPT_TEMPLATE = (
    "[MATH_TASK] Problem:\n{problem}\n\n"
    "Reference answer: {gold}\n\n"
    "Below is one diverse complete step-by-step solution that ends with "
    'the final answer in \\boxed{{...}}:\n'
)


@dataclass
class AnswerSet:
    problem_id: str
    gold: str
    y: List[str] = field(default_factory=list)

    @property
    def m(self) -> int:
        return len(self.y)


@dataclass
class AnswerSetGenerator:
    """Async builder for Y.

    `sample_fn(prompt, n, temperature, max_tokens)` must return a list of `n`
    completions; it is normally a thin wrapper over the guidance vLLM client.
    """

    sample_fn: Callable[..., Awaitable[List[str]]]
    m: int = 100
    temperature: float = 0.7
    max_tokens: int = 512
    prompt_template: str = DEFAULT_Y_PROMPT_TEMPLATE
    min_chars: int = 8

    async def build(self, problem_id: str, problem: str, gold: str) -> AnswerSet:
        prompt = self.prompt_template.format(problem=problem, gold=gold)
        completions = await self.sample_fn(
            prompt=prompt,
            n=self.m,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # Drop empties; keep the order so logging is stable.
        completions = [c for c in completions if c and len(c) >= self.min_chars]
        return AnswerSet(problem_id=problem_id, gold=gold, y=completions)
