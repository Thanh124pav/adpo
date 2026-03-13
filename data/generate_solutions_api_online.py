"""
Generate reference solutions using ONLINE reasoning model APIs.

Supports multiple providers optimized for math reasoning:
  - DeepSeek (R1, R1-0528)
  - OpenAI (o1, o3-mini, o4-mini)
  - Together AI (Qwen3, DeepSeek-R1)
  - Google Gemini (2.5 Pro/Flash thinking)

Key differences from generate_solutions_api.py (local endpoint):
  - Requires API key (via --api_key or env var)
  - Provider-specific handling (system msg support, temperature, reasoning tokens)
  - Presets for common providers (--provider deepseek)
  - Extracts reasoning_content from models that return it separately

Usage:
    # DeepSeek-R1
    python data/generate_solutions_api_online.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions_r1.jsonl \
        --provider deepseek \
        --strategy both

    # OpenAI o3-mini
    python data/generate_solutions_api_online.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions_o3.jsonl \
        --provider openai --model o3-mini \
        --strategy hint

    # Custom endpoint with API key
    python data/generate_solutions_api_online.py \
        --input data/processed/train/math.parquet \
        --output data/solutions/math_solutions.jsonl \
        --endpoint https://api.example.com \
        --model my-model \
        --api_key sk-xxx \
        --strategy free
"""

import argparse
import asyncio
import json
import os
import re
import hashlib
import logging
import time
from typing import List, Dict, Optional, Tuple

import pandas as pd
import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDER_PRESETS = {
    "deepseek": {
        "endpoint": "https://api.deepseek.com",
        "model": "deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
        "supports_system": True,
        "supports_temperature": False,  # R1 ignores temperature
        "has_reasoning_content": True,
        "max_concurrent": 16,  # conservative for paid API
        "notes": "DeepSeek-R1; also try deepseek-chat for V3",
    },
    "openai": {
        "endpoint": "https://api.openai.com",
        "model": "o3-mini",
        "env_key": "OPENAI_API_KEY",
        "supports_system": False,  # o-series: system msg → developer msg
        "supports_temperature": False,  # o-series ignores temperature
        "has_reasoning_content": False,
        "use_developer_role": True,  # o-series uses "developer" instead of "system"
        "max_concurrent": 16,
        "notes": "o1, o1-mini, o3-mini, o4-mini; for gpt-4o set supports_system/temperature=True",
    },
    "together": {
        "endpoint": "https://api.together.xyz",
        "model": "deepseek-ai/DeepSeek-R1",
        "env_key": "TOGETHER_API_KEY",
        "supports_system": True,
        "supports_temperature": True,
        "has_reasoning_content": False,  # Together returns reasoning inline
        "max_concurrent": 32,
        "notes": "Also: Qwen/Qwen3-235B-A22B, meta-llama/Llama-4-Maverick-17B-128E",
    },
    "google": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-pro",
        "env_key": "GOOGLE_API_KEY",
        "supports_system": True,
        "supports_temperature": True,
        "has_reasoning_content": False,
        "max_concurrent": 8,  # Gemini rate limits are strict
        "notes": "gemini-2.5-pro, gemini-2.5-flash (thinking models)",
    },
    "openrouter": {
        "endpoint": "https://openrouter.ai/api",
        "model": "deepseek/deepseek-r1",
        "env_key": "OPENROUTER_API_KEY",
        "supports_system": True,
        "supports_temperature": True,
        "has_reasoning_content": True,
        "max_concurrent": 24,
        "notes": "Access many models; also: openai/o3-mini, qwen/qwen3-235b-a22b",
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HINT_SYSTEM_PROMPT = (
    "You are a math problem solver. You are given a math problem and its correct answer. "
    "Write a detailed step-by-step solution that arrives at the given answer.\n\n"
    "IMPORTANT formatting rules:\n"
    "- Use '.' ONLY at the very end of your complete solution to mark the finish\n"
    "- Do NOT use '.' anywhere else in your reasoning (use commas, semicolons, "
    "colons, or line breaks instead)\n"
    "- Put your final answer within \\boxed{}"
)

FREE_SYSTEM_PROMPT = (
    "Please reason step by step to solve the following math problem.\n\n"
    "IMPORTANT formatting rules:\n"
    "- Use '.' ONLY at the very end of your complete solution to mark the finish\n"
    "- Do NOT use '.' anywhere else in your reasoning (use commas, semicolons, "
    "colons, or line breaks instead)\n"
    "- Put your final answer within \\boxed{}"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_question_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()[:16]


def extract_question_from_prompt(prompt_json: str) -> str:
    try:
        messages = json.loads(prompt_json)
        for msg in messages:
            if msg["role"] == "user":
                return msg["content"]
    except (json.JSONDecodeError, KeyError):
        pass
    return prompt_json


def extract_ground_truth(reward_model_json: str) -> str:
    try:
        data = json.loads(reward_model_json)
        return data.get("ground_truth", "")
    except (json.JSONDecodeError, KeyError):
        return reward_model_json


def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def normalize_answer(answer: str) -> str:
    answer = answer.strip().replace(" ", "")
    for cmd in ["\\frac", "\\dfrac"]:
        answer = answer.replace(cmd, "frac")
    for cmd in ["\\left", "\\right", "\\,"]:
        answer = answer.replace(cmd, "")
    return answer.lower()


def check_answer_correct(solution: str, ground_truth: str) -> bool:
    extracted = extract_boxed_answer(solution)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(ground_truth)


# ---------------------------------------------------------------------------
# API Client (online, with auth)
# ---------------------------------------------------------------------------

class OnlineAPIClient:
    """Async client for online reasoning model APIs with auth + provider quirks."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        max_concurrent: int = 16,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        timeout: float = 300.0,
        supports_system: bool = True,
        supports_temperature: bool = True,
        has_reasoning_content: bool = False,
        use_developer_role: bool = False,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.completions_url = f"{self.endpoint}/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.supports_system = supports_system
        self.supports_temperature = supports_temperature
        self.has_reasoning_content = has_reasoning_content
        self.use_developer_role = use_developer_role
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._stats = {
            "total": 0, "success": 0, "failed": 0, "retries": 0,
            "reasoning_tokens": 0, "completion_tokens": 0,
        }

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _adapt_messages(self, messages: List[dict]) -> List[dict]:
        """Adapt messages for provider quirks."""
        adapted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                if not self.supports_system and self.use_developer_role:
                    # OpenAI o-series: system → developer
                    adapted.append({"role": "developer", "content": content})
                elif not self.supports_system:
                    # Prepend system content to first user message
                    continue  # handled below
                else:
                    adapted.append(msg)
            else:
                adapted.append(msg)

        # If system msg not supported and no developer role, merge into user msg
        if not self.supports_system and not self.use_developer_role:
            system_parts = [m["content"] for m in messages if m["role"] == "system"]
            if system_parts:
                system_text = "\n".join(system_parts)
                for i, msg in enumerate(adapted):
                    if msg["role"] == "user":
                        adapted[i] = {
                            "role": "user",
                            "content": f"{system_text}\n\n{msg['content']}",
                        }
                        break

        return adapted

    def _extract_response(self, data: dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract (content, reasoning_content) from API response."""
        choice = data["choices"][0]["message"]
        content = choice.get("content", "")
        reasoning = choice.get("reasoning_content")  # DeepSeek-R1 specific

        # Track token usage
        usage = data.get("usage", {})
        self._stats["completion_tokens"] += usage.get("completion_tokens", 0)
        if "completion_tokens_details" in usage:
            self._stats["reasoning_tokens"] += usage["completion_tokens_details"].get(
                "reasoning_tokens", 0
            )

        return content, reasoning

    async def _single_request(
        self,
        session: aiohttp.ClientSession,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Optional[Dict[str, str]]:
        """Single request with retries. Returns {"content": ..., "reasoning": ...}."""
        adapted = self._adapt_messages(messages)

        payload = {
            "model": self.model,
            "messages": adapted,
            "max_completion_tokens": max_tokens,
        }
        if self.supports_temperature:
            payload["temperature"] = temperature
            payload["top_p"] = 0.95

        headers = self._build_headers()

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    self._stats["total"] += 1
                    async with session.post(
                        self.completions_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content, reasoning = self._extract_response(data)
                            self._stats["success"] += 1
                            return {
                                "content": content or "",
                                "reasoning": reasoning,
                            }
                        elif resp.status == 429:
                            self._stats["retries"] += 1
                            wait = self.retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limited, waiting {wait:.1f}s...")
                            await asyncio.sleep(wait)
                        elif resp.status in (500, 502, 503):
                            self._stats["retries"] += 1
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        else:
                            body = await resp.text()
                            logger.warning(f"API error {resp.status}: {body[:300]}")
                            self._stats["retries"] += 1
                            await asyncio.sleep(self.retry_delay)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Request error (attempt {attempt+1}): {e}")
                self._stats["retries"] += 1
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

        self._stats["failed"] += 1
        return None

    async def generate_batch(
        self,
        all_messages: List[List[dict]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> List[Optional[Dict[str, str]]]:
        """Send batch concurrently. Returns list of {"content", "reasoning"}."""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._single_request(session, msgs, temperature, max_tokens)
                for msgs in all_messages
            ]
            return list(await asyncio.gather(*tasks))

    def get_stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_hint_messages(question: str, answer: str) -> List[dict]:
    return [
        {"role": "system", "content": HINT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"**Problem:** {question}\n\n"
            f"**Correct Answer:** {answer}\n\n"
            "Write a detailed step-by-step solution that arrives at this answer"
        )},
    ]


def build_free_messages(question: str) -> List[dict]:
    return [
        {"role": "system", "content": FREE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Generation Logic
# ---------------------------------------------------------------------------

def _get_solution_text(result: Optional[Dict[str, str]], include_reasoning: bool) -> Optional[str]:
    """Combine reasoning + content into a single solution string."""
    if result is None:
        return None
    content = result["content"] or ""
    reasoning = result.get("reasoning")
    if include_reasoning and reasoning:
        # Models like DeepSeek-R1 return reasoning separately
        # Combine: <think>reasoning</think>\n\ncontent
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    return content


async def generate_hint_solutions(
    client: OnlineAPIClient,
    questions: List[str],
    answers: List[str],
    num_solutions: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    include_reasoning: bool = True,
) -> List[List[str]]:
    """Generate solutions with hint (answer given). All kept."""
    all_messages = []
    index_map = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        for _ in range(num_solutions):
            all_messages.append(build_hint_messages(q, a))
            index_map.append(i)

    logger.info(f"Sending {len(all_messages)} hint requests...")
    results = await client.generate_batch(all_messages, temperature, max_tokens)

    solutions = [[] for _ in questions]
    for qi, result in zip(index_map, results):
        text = _get_solution_text(result, include_reasoning)
        if text:
            solutions[qi].append(text)

    return solutions


async def generate_free_solutions(
    client: OnlineAPIClient,
    questions: List[str],
    ground_truths: List[str],
    num_solutions: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    include_reasoning: bool = True,
) -> List[List[str]]:
    """Generate solutions freely, keep only correct ones."""
    all_messages = []
    index_map = []

    for i, q in enumerate(questions):
        for _ in range(num_solutions):
            all_messages.append(build_free_messages(q))
            index_map.append(i)

    logger.info(f"Sending {len(all_messages)} free requests...")
    results = await client.generate_batch(all_messages, temperature, max_tokens)

    solutions = [[] for _ in questions]
    total_generated = 0
    total_correct = 0

    for qi, result in zip(index_map, results):
        text = _get_solution_text(result, include_reasoning)
        if text is None:
            continue
        total_generated += 1
        # Check correctness against the content part (where \boxed{} lives)
        check_text = (result["content"] or text) if result else text
        if check_answer_correct(check_text, ground_truths[qi]):
            solutions[qi].append(text)
            total_correct += 1

    logger.info(
        f"Free generation: {total_correct}/{total_generated} correct "
        f"({100*total_correct/max(total_generated,1):.1f}%)"
    )
    return solutions


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

async def run_generation(args):
    # Resolve provider preset
    preset = {}
    if args.provider:
        if args.provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown provider '{args.provider}'. "
                f"Available: {list(PROVIDER_PRESETS.keys())}"
            )
        preset = PROVIDER_PRESETS[args.provider]
        print(f"Provider: {args.provider} — {preset['notes']}")

    endpoint = args.endpoint or preset.get("endpoint", "https://api.openai.com")
    model = args.model or preset.get("model", "o3-mini")
    max_concurrent = args.max_concurrent or preset.get("max_concurrent", 16)

    # Resolve API key: --api_key > env var from preset > OPENAI_API_KEY fallback
    api_key = args.api_key
    if not api_key:
        env_key = preset.get("env_key", "OPENAI_API_KEY")
        api_key = os.environ.get(env_key) or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            f"API key required. Set --api_key, or env var "
            f"{preset.get('env_key', 'OPENAI_API_KEY')}"
        )

    supports_system = preset.get("supports_system", True)
    supports_temperature = preset.get("supports_temperature", True)
    has_reasoning_content = preset.get("has_reasoning_content", False)
    use_developer_role = preset.get("use_developer_role", False)

    # CLI overrides
    if args.no_system_msg:
        supports_system = False
    if args.no_temperature:
        supports_temperature = False

    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print(f"Concurrency: {max_concurrent}")
    print(f"System msg: {supports_system} | Temperature: {supports_temperature} "
          f"| Reasoning content: {has_reasoning_content}")

    # Load dataset
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} questions from {args.input}")

    questions = [extract_question_from_prompt(r) for r in df["prompt"]]
    ground_truths = [extract_ground_truth(r) for r in df["reward_model"]]
    data_sources = list(df["data_source"])

    # Skip existing
    existing_ids = set()
    if args.append and os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["question_id"])
        print(f"Skipping {len(existing_ids)} already-processed questions")

    new_indices = [
        i for i in range(len(questions))
        if make_question_id(questions[i]) not in existing_ids
    ]

    if not new_indices:
        print("All questions already processed")
        return

    print(f"{len(new_indices)} questions to process")

    # Create client
    client = OnlineAPIClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        supports_system=supports_system,
        supports_temperature=supports_temperature,
        has_reasoning_content=has_reasoning_content,
        use_developer_role=use_developer_role,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.append else "w"

    total_written = 0
    total_solutions = 0
    t_start = time.time()

    for batch_start in range(0, len(new_indices), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(new_indices))
        batch_idx = new_indices[batch_start:batch_end]

        batch_questions = [questions[i] for i in batch_idx]
        batch_answers = [ground_truths[i] for i in batch_idx]

        batch_num = batch_start // args.batch_size + 1
        print(f"\n--- Batch {batch_num}: {len(batch_idx)} questions ---")

        if args.strategy == "hint":
            batch_solutions = await generate_hint_solutions(
                client, batch_questions, batch_answers,
                num_solutions=args.num_solutions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                include_reasoning=has_reasoning_content,
            )
        elif args.strategy == "free":
            batch_solutions = await generate_free_solutions(
                client, batch_questions, batch_answers,
                num_solutions=args.num_solutions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                include_reasoning=has_reasoning_content,
            )
        elif args.strategy == "both":
            hint_sols = await generate_hint_solutions(
                client, batch_questions, batch_answers,
                num_solutions=max(1, args.num_solutions // 2),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                include_reasoning=has_reasoning_content,
            )
            free_sols = await generate_free_solutions(
                client, batch_questions, batch_answers,
                num_solutions=args.num_solutions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                include_reasoning=has_reasoning_content,
            )
            batch_solutions = [h + f for h, f in zip(hint_sols, free_sols)]
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        # Write
        with open(args.output, mode) as f:
            for j, idx in enumerate(batch_idx):
                sols = batch_solutions[j]
                if not sols:
                    continue
                record = {
                    "question_id": make_question_id(questions[idx]),
                    "question": questions[idx],
                    "ground_truth": ground_truths[idx],
                    "data_source": data_sources[idx],
                    "solutions": sols,
                    "strategy": args.strategy,
                    "model": model,
                    "provider": args.provider or "custom",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                total_solutions += len(sols)

        mode = "a"
        elapsed = time.time() - t_start
        rate = total_written / max(elapsed, 1)
        stats = client.get_stats()
        print(f"  Written: {total_written} questions, {total_solutions} solutions "
              f"({rate:.1f} q/s)")
        print(f"  API: {stats['success']} ok, {stats['failed']} fail, "
              f"{stats['retries']} retries")
        if stats["reasoning_tokens"] > 0:
            print(f"  Tokens: {stats['reasoning_tokens']} reasoning, "
                  f"{stats['completion_tokens']} completion")

    elapsed = time.time() - t_start
    stats = client.get_stats()
    print(f"\nDone! {total_written} questions, {total_solutions} solutions "
          f"-> {args.output}")
    print(f"Time: {elapsed:.0f}s | Final API stats: {stats}")

    # Cost estimate (rough)
    if stats["completion_tokens"] > 0:
        print(f"\nToken usage: ~{stats['completion_tokens']:,} completion tokens")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate math solutions via online reasoning model APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Provider presets (--provider):
  deepseek   DeepSeek-R1 (deepseek-reasoner)        $0.55/M output
  openai     OpenAI o3-mini                          $4.40/M output
  together   Together AI (DeepSeek-R1, Qwen3)        $2.50/M output
  google     Gemini 2.5 Pro (thinking)               $10/M output
  openrouter OpenRouter (many models)                varies

Examples:
  %(prog)s --input data.parquet --output sols.jsonl --provider deepseek --strategy both
  %(prog)s --input data.parquet --output sols.jsonl --provider openai --model o4-mini
  %(prog)s --input data.parquet --output sols.jsonl --endpoint https://custom.api --api_key sk-xxx
        """,
    )

    # I/O
    parser.add_argument("--input", type=str, required=True,
                        help="Input parquet file (verl format)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file")

    # Provider / connection
    parser.add_argument("--provider", type=str, default=None,
                        choices=list(PROVIDER_PRESETS.keys()),
                        help="Use a provider preset (sets endpoint, model, quirks)")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="API endpoint URL (overrides provider preset)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides provider preset)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set env var per provider)")

    # Strategy
    parser.add_argument("--strategy", type=str, default="hint",
                        choices=["hint", "free", "both"],
                        help="hint: answer given; free: verify; both: mix")
    parser.add_argument("--num_solutions", type=int, default=4,
                        help="Solutions per question")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (ignored if model doesn't support)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max completion tokens (reasoning models need more)")

    # Connection params
    parser.add_argument("--max_concurrent", type=int, default=None,
                        help="Max concurrent requests (default: from provider preset)")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=2.0,
                        help="Base retry delay in seconds")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Request timeout (reasoning models can be slow)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Questions per batch")

    # Behavior overrides
    parser.add_argument("--no_system_msg", action="store_true",
                        help="Force disable system messages")
    parser.add_argument("--no_temperature", action="store_true",
                        help="Force disable temperature parameter")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output, skip processed questions")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
