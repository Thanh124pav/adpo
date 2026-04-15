"""
Trajectory Stitching -- splice golden paths onto wrong responses.

When ALL n responses in a GRPO group are wrong, this module:
1. Segments the golden path into phases (via endpoint log-probs)
2. Finds the optimal splice point (j, i) per wrong response
3. Iteratively rolls out from the stitched prefix with reward decay
4. Produces concentrated advantages at the splice point

Algorithm:
    For each wrong response (segmented into phases 0..J):
      For j from J down to 0:
        For each golden phase i:
          if len(response[:j]) + len(golden[i:]) <= max_len:
            score = log_prob(golden_token_i | response[:j])
            track best (j, i)

      stitched_prefix = response[:j*] + golden[i*]
      rollout from stitched_prefix
        if correct -> reward = 1.0
        if wrong -> extend: response[:j*] + golden[i*, i*+1], reward = 0.9
        continue until correct or exhausted (reward decays each step)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


@dataclass
class SpliceResult:
    """Result of trajectory stitching for one response."""
    response_idx: int           # index in batch
    splice_j: int               # response phase index (cut after phase j)
    splice_i: int               # golden phase index (graft from phase i)
    splice_token_pos: int       # token position in response where splice happens
    stitched_text: str          # full stitched + rolled out text
    reward: float               # reward (1.0, 0.9, 0.8, ... or 0 if failed)
    golden_phases_used: int     # how many golden phases were needed
    verified: bool              # whether the stitched response is correct


class TrajectoryStitcher:
    """Splice golden paths onto wrong responses and rollout.

    Uses an OpenAI-compatible endpoint for:
    - Computing log-probs of golden path tokens (phase segmentation)
    - Computing P(golden[i] | response[:j]) for splice point finding
    - Rolling out continuations from stitched prefixes
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        tokenizer,
        max_response_length: int = 2048,
        max_concurrent: int = 32,
        timeout: float = 120.0,
        reward_decay: float = 0.1,
        max_golden_extensions: int = 5,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.completions_url = f"{self.endpoint}/v1/completions"
        self.model = model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.reward_decay = reward_decay
        self.max_golden_extensions = max_golden_extensions

    # ------------------------------------------------------------------
    # Endpoint helpers
    # ------------------------------------------------------------------

    async def _completions_batch(
        self, session, semaphore, prompts: List[str],
        max_tokens: int = 0, echo: bool = True, logprobs: int = 1,
        temperature: float = 0.0,
    ) -> List[Optional[dict]]:
        """Batch completion requests to the endpoint."""
        results = [None] * len(prompts)

        async def _request(idx: int, prompt: str):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "echo": echo,
                "logprobs": logprobs,
            }
            for attempt in range(3):
                try:
                    async with semaphore:
                        async with session.post(
                            self.completions_url, json=payload,
                            timeout=__import__('aiohttp').ClientTimeout(
                                total=self.timeout
                            ),
                        ) as resp:
                            data = await resp.json()
                            results[idx] = data
                            return
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Completion request failed: {e}")

        import aiohttp
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(self.max_concurrent)
            tasks = [_request(i, p) for i, p in enumerate(prompts)]
            await asyncio.gather(*tasks)

        return results

    async def _generate_batch(
        self, session, semaphore, prompts: List[str],
        max_tokens: int = 2048, temperature: float = 0.7,
    ) -> List[str]:
        """Generate completions from prompts."""
        results = [""] * len(prompts)

        async def _request(idx: int, prompt: str):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
            }
            for attempt in range(3):
                try:
                    async with semaphore:
                        async with session.post(
                            self.completions_url, json=payload,
                            timeout=__import__('aiohttp').ClientTimeout(
                                total=self.timeout
                            ),
                        ) as resp:
                            data = await resp.json()
                            text = data["choices"][0]["text"]
                            results[idx] = text
                            return
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Generate request failed: {e}")

        import aiohttp
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(self.max_concurrent)
            tasks = [_request(i, p) for i, p in enumerate(prompts)]
            await asyncio.gather(*tasks)

        return results

    def _run_async(self, coro):
        """Run async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    # ------------------------------------------------------------------
    # Golden path phase segmentation
    # ------------------------------------------------------------------

    def segment_golden_path(
        self, golden_text: str, question: str,
    ) -> Tuple[List[str], List[int]]:
        """Segment golden path into phases using sentence boundaries.

        Returns:
            phase_texts: list of phase text strings
            phase_char_offsets: character offsets where each phase starts
        """
        # Split on sentence-ending periods followed by whitespace/newline,
        # preserving step structure from the golden path
        import re

        # Split on common math solution step patterns
        # Priority: explicit step markers, then double newlines, then sentences
        segments = []
        current = []
        char_offsets = [0]

        lines = golden_text.split('\n')
        current_offset = 0

        for line in lines:
            stripped = line.strip()
            # Check if this is a new step/paragraph
            is_step_start = bool(re.match(
                r'^(Step\s+\d|\\text\{Step|First|Second|Third|Next|Then|'
                r'Finally|Therefore|Thus|Hence|So,|Now,|Since|Let|We\s+have|'
                r'We\s+know|We\s+can|Note\s+that|\d+[\.\)])',
                stripped, re.IGNORECASE
            ))

            if is_step_start and current:
                segments.append('\n'.join(current))
                char_offsets.append(current_offset)
                current = []

            current.append(line)
            current_offset += len(line) + 1  # +1 for \n

        if current:
            segments.append('\n'.join(current))

        # If we got too few segments, split by sentences
        if len(segments) < 2:
            segments = []
            char_offsets = [0]
            # Split by sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?])\s+', golden_text)
            offset = 0
            for sent in sentences:
                if sent.strip():
                    segments.append(sent)
                    char_offsets.append(offset + len(sent) + 1)
                    offset += len(sent) + 1

        # Merge very short segments
        min_len = 20
        merged = []
        merged_offsets = [0]
        buffer = ""
        for seg, off in zip(segments, char_offsets):
            if buffer:
                buffer += " " + seg
            else:
                buffer = seg
            if len(buffer) >= min_len:
                merged.append(buffer)
                merged_offsets.append(off + len(seg))
                buffer = ""
        if buffer:
            if merged:
                merged[-1] += " " + buffer
            else:
                merged.append(buffer)

        return merged if merged else [golden_text], merged_offsets[:len(merged) + 1]

    # ------------------------------------------------------------------
    # Splice point finding
    # ------------------------------------------------------------------

    def find_splice_points(
        self,
        response_phase_texts: List[List[str]],
        golden_phase_texts: List[str],
        max_response_length: int,
    ) -> List[Tuple[int, int, float]]:
        """Find optimal (j, i) splice point for each wrong response.

        For each response, iterates j from high to low and finds the golden
        phase i that maximizes P(golden[i] | response[:j]).

        Args:
            response_phase_texts: [n_responses][n_phases] phase texts
            golden_phase_texts: [n_golden_phases] phase texts
            max_response_length: max character length constraint

        Returns:
            List of (j, i, log_prob) per response. (-1, -1, -inf) if no valid splice.
        """
        n_responses = len(response_phase_texts)
        n_golden = len(golden_phase_texts)

        # Build all (response_idx, j, i, prefix_text, probe_text) candidates
        candidates = []
        for r in range(n_responses):
            phases = response_phase_texts[r]
            # j from high to low (prefer keeping more original reasoning)
            for j in range(len(phases) - 1, -1, -1):
                prefix = " ".join(phases[:j + 1])
                for i in range(n_golden):
                    golden_suffix = " ".join(golden_phase_texts[i:])
                    total_len = len(prefix) + len(golden_suffix)
                    if total_len > max_response_length:
                        continue
                    # Probe: first ~50 chars of golden phase i
                    probe = golden_phase_texts[i][:200]
                    candidates.append((r, j, i, prefix, probe))

        if not candidates:
            return [(-1, -1, float('-inf'))] * n_responses

        # Batch compute log_probs: P(probe | prefix) via echo + logprobs
        prompts = [f"{prefix} {probe}" for (_, _, _, prefix, probe) in candidates]
        prefix_lens = [len(c[3]) + 1 for c in candidates]  # +1 for space

        async def _compute():
            import aiohttp
            sem = asyncio.Semaphore(self.max_concurrent)
            async with aiohttp.ClientSession() as session:
                return await self._completions_batch(
                    session, sem, prompts,
                    max_tokens=0, echo=True, logprobs=1,
                )

        results = self._run_async(_compute())

        # Extract log_probs of the probe portion (tokens after prefix)
        splice_scores: Dict[int, Tuple[int, int, float]] = {}

        for idx, ((r, j, i, prefix, probe), result) in enumerate(
            zip(candidates, results)
        ):
            if result is None:
                continue

            try:
                choice = result["choices"][0]
                token_logprobs = choice["logprobs"]["token_logprobs"]
                text_offsets = choice["logprobs"]["text_offset"]

                # Find where the probe starts (after prefix)
                prefix_char_len = len(prefix) + 1  # +1 for space
                probe_start_idx = 0
                for t, offset in enumerate(text_offsets):
                    if offset >= prefix_char_len:
                        probe_start_idx = t
                        break

                # Average log_prob of probe tokens
                probe_logprobs = [
                    lp for lp in token_logprobs[probe_start_idx:]
                    if lp is not None
                ]
                if probe_logprobs:
                    avg_logprob = sum(probe_logprobs) / len(probe_logprobs)
                else:
                    avg_logprob = float('-inf')

                # Keep best (j, i) per response
                if r not in splice_scores or avg_logprob > splice_scores[r][2]:
                    splice_scores[r] = (j, i, avg_logprob)

            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Failed to parse logprobs for candidate {idx}: {e}")
                continue

        # Build result list
        out = []
        for r in range(n_responses):
            if r in splice_scores:
                out.append(splice_scores[r])
            else:
                out.append((-1, -1, float('-inf')))

        return out

    # ------------------------------------------------------------------
    # Iterative rollout with reward decay
    # ------------------------------------------------------------------

    def iterative_rollout(
        self,
        response_phase_texts: List[str],
        golden_phase_texts: List[str],
        splice_j: int,
        splice_i: int,
        question: str,
        answer: str,
        data_source: str,
    ) -> SpliceResult:
        """Roll out from stitched prefix, extending golden phases if needed.

        Flow:
            prefix = response[:j+1] + golden[i]
            rollout → check answer
            if wrong: prefix = response[:j+1] + golden[i:i+2], reward *= decay
            repeat until correct or max extensions reached

        Returns:
            SpliceResult with the best stitched response found.
        """
        reward_if_correct = 1.0
        prefix_base = " ".join(response_phase_texts[:splice_j + 1])
        max_remaining = self.max_response_length - len(prefix_base)

        for ext in range(self.max_golden_extensions):
            golden_end = min(splice_i + 1 + ext, len(golden_phase_texts))
            golden_chunk = " ".join(golden_phase_texts[splice_i:golden_end])

            if len(golden_chunk) > max_remaining:
                break

            stitched_prefix = f"{prefix_base} {golden_chunk}"
            remaining_tokens = max(
                256,
                (self.max_response_length - len(stitched_prefix)) // 4,
            )

            # Rollout from stitched prefix
            async def _rollout():
                import aiohttp
                sem = asyncio.Semaphore(self.max_concurrent)
                async with aiohttp.ClientSession() as session:
                    return await self._generate_batch(
                        session, sem, [stitched_prefix],
                        max_tokens=remaining_tokens, temperature=0.7,
                    )

            continuations = self._run_async(_rollout())
            full_text = stitched_prefix + continuations[0]

            # Verify
            score = compute_score(
                data_source=data_source,
                solution_str=full_text,
                ground_truth=answer,
            )

            if score >= 1.0:
                return SpliceResult(
                    response_idx=-1,  # will be set by caller
                    splice_j=splice_j,
                    splice_i=splice_i,
                    splice_token_pos=-1,  # will be set later
                    stitched_text=full_text,
                    reward=reward_if_correct,
                    golden_phases_used=golden_end - splice_i,
                    verified=True,
                )

            reward_if_correct -= self.reward_decay
            if reward_if_correct <= 0:
                break

        # All attempts failed
        return SpliceResult(
            response_idx=-1,
            splice_j=splice_j,
            splice_i=splice_i,
            splice_token_pos=-1,
            stitched_text="",
            reward=0.0,
            golden_phases_used=0,
            verified=False,
        )

    # ------------------------------------------------------------------
    # Full stitching pipeline
    # ------------------------------------------------------------------

    def stitch_group(
        self,
        questions: List[str],
        response_phase_texts: List[List[str]],
        golden_path: str,
        golden_answers: List[str],
        data_sources: List[str],
        response_boundaries: List[List[int]],
    ) -> List[SpliceResult]:
        """Run trajectory stitching for one all-wrong group.

        Args:
            questions: [n] question texts (same for all in group)
            response_phase_texts: [n][K] phase texts per response
            golden_path: golden solution text
            golden_answers: [n] ground truth answers
            data_sources: [n] dataset names
            response_boundaries: [n][K+1] token boundaries per response

        Returns:
            [n] SpliceResults, one per response in the group.
        """
        n = len(response_phase_texts)

        # Step 1: Segment golden path into phases
        golden_phases, golden_offsets = self.segment_golden_path(
            golden_path, questions[0]
        )
        logger.info(
            f"[Stitch] Golden path: {len(golden_phases)} phases, "
            f"lengths={[len(p) for p in golden_phases]}"
        )

        # Step 2: Find splice points for all responses
        splice_points = self.find_splice_points(
            response_phase_texts, golden_phases, self.max_response_length,
        )

        # Step 3: Iterative rollout per response
        results = []
        for r in range(n):
            j, i, log_prob = splice_points[r]

            if j < 0 or i < 0:
                results.append(SpliceResult(
                    response_idx=r, splice_j=-1, splice_i=-1,
                    splice_token_pos=-1, stitched_text="",
                    reward=0.0, golden_phases_used=0, verified=False,
                ))
                continue

            result = self.iterative_rollout(
                response_phase_texts=response_phase_texts[r],
                golden_phase_texts=golden_phases,
                splice_j=j,
                splice_i=i,
                question=questions[r],
                answer=golden_answers[r],
                data_source=data_sources[r],
            )
            result.response_idx = r

            # Map splice_j to token position
            if j + 1 < len(response_boundaries[r]):
                result.splice_token_pos = response_boundaries[r][j + 1]
            elif response_boundaries[r]:
                result.splice_token_pos = response_boundaries[r][-1]

            results.append(result)

        n_verified = sum(1 for r in results if r.verified)
        logger.info(
            f"[Stitch] Group results: {n_verified}/{n} verified correct, "
            f"splice_points={[(r.splice_j, r.splice_i) for r in results]}"
        )

        return results


def compute_stitched_advantages(
    token_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    stitch_results: Dict[int, SpliceResult],
    index: torch.Tensor,
    splice_boost: float = 2.0,
    pre_splice_advantage: float = 0.0,
    post_splice_decay: float = 0.9,
) -> torch.Tensor:
    """Modify token advantages for all-wrong groups based on stitching results.

    For responses in all-wrong groups where stitching succeeded:
    - Tokens before splice point: advantage = pre_splice_advantage (neutral)
    - Tokens at splice point: advantage = splice_boost * reward (strong positive)
    - Tokens after splice point: advantage = -|original| (negative, was wrong)

    The concentrated positive advantage at the splice point teaches the model
    the critical decision point where it should have diverged to the correct path.

    Args:
        token_advantages: (batch, seq_len) original ADPO advantages
        response_mask: (batch, seq_len) response token mask
        stitch_results: {batch_idx: SpliceResult} for all-wrong responses
        index: (batch,) group UIDs
        splice_boost: multiplier for advantage at splice point
        pre_splice_advantage: advantage for tokens before splice (default 0)
        post_splice_decay: geometric decay for positive signal after splice

    Returns:
        Modified token_advantages tensor.
    """
    if not stitch_results:
        return token_advantages

    advantages = token_advantages.clone()

    for b, result in stitch_results.items():
        if not result.verified or result.splice_token_pos < 0:
            continue

        splice_pos = result.splice_token_pos
        seq_len = advantages.shape[1]
        reward = result.reward

        # Tokens before splice: neutral (model already generates this well)
        advantages[b, :splice_pos] = pre_splice_advantage * response_mask[b, :splice_pos]

        # Tokens at and after splice point: concentrated positive spike + decay
        active_after = response_mask[b, splice_pos:].nonzero(as_tuple=True)[0]
        if len(active_after) > 0:
            for t_idx, t in enumerate(active_after):
                pos = splice_pos + t.item()
                if pos >= seq_len:
                    break
                # Spike at splice, then geometric decay
                decay_factor = post_splice_decay ** t_idx
                advantages[b, pos] = splice_boost * reward * decay_factor

    return advantages
