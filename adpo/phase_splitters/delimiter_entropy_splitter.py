import torch
import re
import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .base import PhaseSplitter

class DeliEntropySplitter(PhaseSplitter):
    def __init__(self, config):
        super().__init__()
        self.method = config.phase_method
        self.delta = config.phase_delta
        self.percentile = config.phase_percentile
        self.min_phase_len = config.phase_min_len
        self.max_phases = config.phase_max_K 

    def split(
        self,
        entropy: torch.Tensor,
        response_mask: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        tokenizer=None,
    ) -> List[List[int]]:
        """Unified boundary detection dispatcher.

        Methods:
            adaptive  - Sentence-based + mean -log pi of first 1/3 tokens.
            entropy   - Sentence-based + mean entropy of first 1/3 tokens.
        """
        if self.method == "entropy":
            return detect_phase_boundaries_entropy(
                entropy, response_mask, percentile=self.percentile,
                min_phase_len=self.min_phase_len, max_phases=self.max_phases,
                token_ids=token_ids, tokenizer=tokenizer,
            )
        elif self.method == "logprob":
            if log_probs is None:
                neg_log_probs = -1 * entropy
            else:
                neg_log_probs = -1 * log_probs
            return detect_phase_boundaries_logprob(
                neg_log_probs, response_mask, percentile=self.percentile,
                min_phase_len=self.min_phase_len, max_phases=self.max_phases,
                token_ids=token_ids, tokenizer=tokenizer,
            )

        else:
            raise ValueError(f"Unknown boundary method: {self.method}")


def detect_phase_boundaries_logprob(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 85.0,
    min_phase_len: int = 5,
    max_phases: int = 10,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[List[int]]:
    """Detect phase boundaries: find </think>, split thinking into phases, output = 1 phase.

    Algorithm:
    1. Find </think> position → split response into [think_region, output_region]
    2. Within think_region only: split into sentences, compute mean -log_pi
       of first 1/3 tokens per sentence, select top-K above percentile threshold
    3. Output region = 1 single phase (last phase)

    Falls back to peak-based detection if token_ids/tokenizer not available.
    """
    if token_ids is None or tokenizer is None:
        return _detect_phase_boundaries_peak(
            neg_log_probs, response_mask, percentile, min_phase_len, max_phases,
        )

    batch_size = neg_log_probs.shape[0]
    think_ends = _find_think_boundary(token_ids, response_mask, tokenizer)
    all_boundaries = []

    for b in range(batch_size):
        nlp = neg_log_probs[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1
        think_end = think_ends[b]

        # Determine think region
        if think_end is not None and think_end > start:
            think_start = start
            think_end_pos = think_end  # token after </think>
        else:
            # No </think> → entire response is "thinking"
            think_start = start
            think_end_pos = end

        # Step 1: Find sentences ONLY within think region
        # Build a temporary mask for just the think region
        think_len = think_end_pos - think_start
        think_ids = token_ids[b, think_start:think_end_pos].tolist()
        delim_ids = _get_delimiter_token_ids(tokenizer)
        break_positions = [i for i, tid in enumerate(think_ids) if tid in delim_ids]

        # Build sentence spans within think region
        raw_sentences = []
        sent_start = 0
        for bp in break_positions:
            sent_end = bp + 1
            if sent_end > sent_start:
                raw_sentences.append((think_start + sent_start, think_start + sent_end))
            sent_start = sent_end
        if sent_start < think_len:
            raw_sentences.append((think_start + sent_start, think_end_pos))

        # Merge short sentences
        sentences = []
        for s_start, s_end in raw_sentences:
            if sentences and (s_start - sentences[-1][0]) < min_phase_len:
                sentences[-1] = (sentences[-1][0], s_end)
            else:
                sentences.append((s_start, s_end))

        if not sentences:
            sentences = [(think_start, think_end_pos)]

        # Step 2: Score sentences and select phase boundaries
        sent_scores = []
        for s_start, s_end in sentences:
            T = s_end - s_start
            head_len = max(1, T // 3)
            head_mean = nlp[s_start:s_start + head_len].mean().item()
            sent_scores.append(head_mean)

        # Select candidates above percentile threshold
        candidates = []
        if len(sent_scores) > 1:
            threshold = np.percentile(sent_scores, percentile)
            for i in range(1, len(sentences)):  # skip first
                if sent_scores[i] > threshold:
                    candidates.append((sent_scores[i], sentences[i][0]))
            candidates.sort(key=lambda x: -x[0])

        # Build boundaries: [start] + think candidates + [think_end (output)]
        boundaries = [start]
        max_think_phases = max_phases - 1 if think_end is not None else max_phases
        for _, s_start in candidates[:max_think_phases - 1]:
            boundaries.append(s_start)

        # Add output phase boundary (after </think>)
        if think_end is not None and think_end > start and think_end < end:
            boundaries.append(think_end)

        boundaries.sort()
        all_boundaries.append(boundaries)

        # Debug for first response
        if b == 0:
            n_think_phases = sum(1 for bd in boundaries if think_end is None or bd < think_end)
            has_output = think_end is not None and think_end in boundaries
            print(f"[ADPO Boundaries] resp=0: {len(sentences)} think_sents, "
                  f"think_end={think_end}, "
                  f"n_think_phases={n_think_phases}, has_output={has_output}, "
                  f"boundaries={boundaries}", flush=True)

    return all_boundaries

def detect_phase_boundaries_entropy(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 80.0,
    min_phase_len: int = 5,
    max_phases: int = 10,
    token_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[List[int]]:
    """Detect phase boundaries: find </think>, split thinking into phases using entropy.

    Algorithm (same as adaptive, but uses entropy instead of -log_pi):
    1. Find </think> position → split response into [think_region, output_region]
    2. Within think_region only: split into sentences, compute mean entropy
       of first 1/3 tokens per sentence, select top-K above percentile threshold
    3. Output region = 1 single phase (last phase)

    Falls back to peak-based detection if token_ids/tokenizer not available.
    """
    if token_ids is None or tokenizer is None:
        return _detect_phase_boundaries_peak(
            entropy, response_mask, percentile, min_phase_len, max_phases,
        )

    batch_size = entropy.shape[0]
    think_ends = _find_think_boundary(token_ids, response_mask, tokenizer)
    all_boundaries = []

    for b in range(batch_size):
        ent = entropy[b]
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1
        think_end = think_ends[b]

        # Determine think region
        if think_end is not None and think_end > start:
            think_end_pos = think_end
        else:
            think_end_pos = end

        # Split sentences ONLY within think region
        think_len = think_end_pos - start
        think_ids = token_ids[b, start:think_end_pos].tolist()
        delim_ids = _get_delimiter_token_ids(tokenizer)
        break_positions = [i for i, tid in enumerate(think_ids) if tid in delim_ids]

        raw_sentences = []
        sent_start = 0
        for bp in break_positions:
            sent_end = bp + 1
            if sent_end > sent_start:
                raw_sentences.append((start + sent_start, start + sent_end))
            sent_start = sent_end
        if sent_start < think_len:
            raw_sentences.append((start + sent_start, think_end_pos))

        # Merge short sentences
        sentences = []
        for s_start, s_end in raw_sentences:
            if sentences and (s_start - sentences[-1][0]) < min_phase_len:
                sentences[-1] = (sentences[-1][0], s_end)
            else:
                sentences.append((s_start, s_end))

        if not sentences:
            sentences = [(start, think_end_pos)]

        # Score sentences by mean entropy of first 1/3 tokens
        sent_scores = []
        for s_start, s_end in sentences:
            T = s_end - s_start
            head_len = max(1, T // 3)
            head_mean = ent[s_start:s_start + head_len].mean().item()
            sent_scores.append(head_mean)

        # Select candidates above percentile threshold
        candidates = []
        if len(sent_scores) > 1:
            threshold = np.percentile(sent_scores, percentile)
            for i in range(1, len(sentences)):
                if sent_scores[i] > threshold:
                    candidates.append((sent_scores[i], sentences[i][0]))
            candidates.sort(key=lambda x: -x[0])

        # Build boundaries: [start] + [think_end] + think candidates
        boundaries = [start]
        if think_end is not None and think_end > start and think_end < end:
            boundaries.append(think_end)

        max_think_phases = max_phases - len(boundaries)
        for _, s_start in candidates[:max_think_phases]:
            if s_start not in boundaries and s_start < (think_end_pos if think_end else end):
                boundaries.append(s_start)

        boundaries.sort()
        all_boundaries.append(boundaries)

        # Debug for first response
        if b == 0:
            n_think_phases = sum(1 for bd in boundaries if think_end is None or bd < think_end)
            has_output = think_end is not None and think_end in boundaries
            print(f"[ADPO Entropy Boundaries] resp=0: {len(sentences)} think_sents, "
                  f"think_end={think_end}, "
                  f"n_think_phases={n_think_phases}, has_output={has_output}, "
                  f"boundaries={boundaries}", flush=True)

    return all_boundaries

def _detect_phase_boundaries_peak(
    neg_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    percentile: float = 85.0,
    min_phase_len: int = 10,
    max_phases: int = 10,
) -> List[List[int]]:
    """Legacy fallback: peak-based detection on -log pi."""
    batch_size = neg_log_probs.shape[0]
    all_boundaries = []

    for b in range(batch_size):
        mask = response_mask[b]
        nlp = neg_log_probs[b]

        active = mask.nonzero(as_tuple=True)[0]
        if len(active) == 0:
            all_boundaries.append([0])
            continue

        start = active[0].item()
        end = active[-1].item() + 1

        active_nlp = nlp[active].cpu().numpy()
        delta = np.percentile(active_nlp, percentile)

        candidates = []
        for t in range(start + 1, end - 1):
            val = nlp[t].item()
            if (val > delta
                    and val >= nlp[t - 1].item()
                    and val >= nlp[t + 1].item()):
                candidates.append((t, val))

        candidates.sort(key=lambda x: -x[1])
        boundaries = [start]
        for t, val in candidates:
            if len(boundaries) >= max_phases:
                break
            if all(abs(t - b_) >= min_phase_len for b_ in boundaries):
                boundaries.append(t)

        boundaries.sort()
        all_boundaries.append(boundaries)

    return all_boundaries

def _find_think_boundary(token_ids, response_mask, tokenizer):
    """Find token index where </think> ends for each response in batch.

    Tries multiple strategies:
    1. Single special token ID lookup
    2. Multi-token encode match
    3. Decode full response and find </think> string, then map back to token position

    Returns list of int or None (one per batch element).
    """
    batch_size = response_mask.shape[0]
    results = []

    # Strategy 1: check if </think> is a single token in vocab
    think_end_single = None
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        tid = tokenizer.convert_tokens_to_ids("</think>")
        if tid is not None and tid != getattr(tokenizer, 'unk_token_id', None):
            think_end_single = tid

    # Strategy 2: encode the marker
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    marker_len = len(think_end_ids)

    # Debug once
    _logged = False

    for b in range(batch_size):
        active = response_mask[b].nonzero(as_tuple=True)[0]
        if len(active) == 0:
            results.append(None)
            continue
        start = active[0].item()
        end = active[-1].item() + 1
        ids = token_ids[b, start:end].tolist()

        found = None

        # Strategy 1: single special token
        if think_end_single is not None:
            for i, tid in enumerate(ids):
                if tid == think_end_single:
                    found = start + i + 1
                    break

        # Strategy 2: multi-token match
        if found is None and marker_len > 0:
            for i in range(len(ids) - marker_len + 1):
                if ids[i:i + marker_len] == think_end_ids:
                    found = start + i + marker_len
                    break

        # Strategy 3: decode and string search (fallback)
        if found is None:
            decoded = tokenizer.decode(token_ids[b, start:end].tolist(), skip_special_tokens=False)
            think_pos = decoded.find("</think>")
            if think_pos >= 0:
                # Map character position back to token position
                # Decode token by token until we pass the character position
                char_count = 0
                for i in range(len(ids)):
                    tok_text = tokenizer.decode([ids[i]], skip_special_tokens=False)
                    char_count += len(tok_text)
                    if char_count > think_pos + len("</think>") - 1:
                        found = start + i + 1
                        break

        # Debug for first response
        if b == 0 and not _logged:
            _logged = True
            print(f"[ADPO Think] single_token_id={think_end_single}, "
                  f"encode_ids={think_end_ids}, "
                  f"found_at={found}, resp_len={end-start}", flush=True)

        results.append(found)
    return results

def _get_delimiter_token_ids(tokenizer) -> set:
    """Build set of token IDs that represent sentence delimiters.

    Scans the tokenizer vocabulary for tokens that ARE delimiters
    (contain sentence-ending punctuation followed by whitespace/newline,
    or are newline tokens themselves).

    Cached per tokenizer instance.
    """
    if not hasattr(tokenizer, '_adpo_delim_ids'):
        _DELIM_PATTERN = re.compile(r'(?:[.?!](?:\s|$))|(?:\.{2,})|(?:\n)')
        delim_ids = set()

        # Scan entire vocab for tokens matching delimiter pattern
        vocab = tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            # Decode the token to get its actual text representation
            # (vocab keys may use special encoding like Ġ for space)
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if _DELIM_PATTERN.search(decoded):
                delim_ids.add(token_id)

        # Remove common false positives (single "." appears in numbers like 3.14)
        # Only keep "." if it's followed by space/newline in the token
        dot_only_ids = set()
        for token_str, token_id in vocab.items():
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if decoded.strip() == ".":
                dot_only_ids.add(token_id)
        # Keep dot-only tokens since they usually end sentences;
        # false positives in "3.14" are acceptable (will be merged by min_sentence_len)

        tokenizer._adpo_delim_ids = delim_ids
        print(f"[ADPO] Built delimiter token set: {len(delim_ids)} tokens", flush=True)
    return tokenizer._adpo_delim_ids