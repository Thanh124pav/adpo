import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from .base import BaseReward
from .trajectory_stitching import TrajectoryStitcher, SpliceResult
from ..phase_splitters.base import PhaseSplitter

logger = logging.getLogger(__name__)


class StitchReward(BaseReward):
    """Reward computer for all-wrong GRPO groups via trajectory stitching.

    For groups where every response is wrong:
      1. Retrieve or generate a golden path (step-by-step solution)
      2. Split golden path into phases using the same PhaseSplitter as responses
      3. Find the optimal splice point (j, i) using HF model log-probs
      4. Roll out from stitched prefix; assign concentrated reward at splice

    For groups with ≥1 correct response: uniform outcome reward per phase.

    Constructor args:
        stitcher:    TrajectoryStitcher — handles rollout via endpoint
        splitter:    PhaseSplitter — splits golden path consistently
        config:      OmegaConf / SimpleNamespace with config.algorithm.*
        hf_model:    Loaded HF model for splice-point scoring (required for
                     find_splice_points_hf). Pass None to fall back to endpoint.
        golden_gen:  GoldenPathGenerator or None.
                     None → dataset already has golden solutions; must be
                            supplied via compute(golden_solutions=[...]).
                     Not-None → answer-only dataset; generator will be called
                                for all-wrong groups.
    """

    def __init__(
        self,
        stitcher: TrajectoryStitcher,
        splitter: PhaseSplitter,
        config,
        hf_model=None,
        tokenizer=None,
        golden_gen=None,
    ):
        super().__init__()
        self.stitcher = stitcher
        self.splitter = splitter
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.golden_gen = golden_gen

        algo = config.algorithm
        self.correct_total = getattr(algo, "correct_total", 1.0)
        self.incorrect_total = getattr(algo, "incorrect_total", -1.0)
        self.splice_boost = getattr(algo, "stitch_splice_boost", 2.0)
        self.pre_splice_adv = getattr(algo, "stitch_pre_splice_adv", 0.0)
        self.post_splice_decay = getattr(algo, "stitch_post_splice_decay", 0.9)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        boundaries_batch: List[List[int]],
        response_mask: torch.Tensor,
        index: torch.Tensor,
        outcome_rewards: List[float],
        phase_texts_batch: List[List[str]],
        questions: List[str],
        golden_answers: List[str],
        data_sources: List[str],
        golden_solutions: Optional[List[Optional[str]]] = None,
        **context,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute phase rewards with stitching for all-wrong groups.

        Args:
            boundaries_batch:  [batch][K] phase boundary token indices
            response_mask:     (batch, seq_len) response token mask
            index:             (batch,) group UIDs
            outcome_rewards:   [batch] correctness scores (1.0 = correct)
            phase_texts_batch: [batch][K] decoded phase text strings
            questions:         [batch] question strings
            golden_answers:    [batch] ground-truth answer strings
            data_sources:      [batch] dataset names
            golden_solutions:  [batch] pre-written golden solutions or None.
                               Required when golden_gen is None.
                               Ignored (overridden by generator) otherwise.

        Returns:
            phase_rewards:  (batch, max_K) float tensor
            phase_mask:     (batch, max_K) binary tensor
            metadata:       {"splice_results": Dict[int, SpliceResult]}
        """
        batch_size, seq_len = response_mask.shape
        max_K = max(len(b) for b in boundaries_batch)
        device = response_mask.device

        # --- Base phase rewards: uniform outcome / n_phases per response ---
        phase_rewards = torch.zeros(batch_size, max_K, device=device)
        phase_mask_t = torch.zeros(batch_size, max_K, device=device)
        for b in range(batch_size):
            n = len(boundaries_batch[b])
            phase_mask_t[b, :n] = 1.0
            base_r = self.correct_total if outcome_rewards[b] >= 1.0 \
                else self.incorrect_total
            phase_rewards[b, :n] = base_r / max(n, 1)

        # --- Find all-wrong groups ---
        all_wrong_uids: Dict[int, List[int]] = {}
        for uid in torch.unique(index).tolist():
            group_idx = [b for b in range(batch_size) if index[b].item() == uid]
            if all(outcome_rewards[b] < 1.0 for b in group_idx):
                all_wrong_uids[uid] = group_idx

        if not all_wrong_uids:
            return phase_rewards, phase_mask_t, {}

        # --- For each all-wrong group: get golden path, split, stitch ---
        splice_results: Dict[int, SpliceResult] = {}

        for uid, group_idx in all_wrong_uids.items():
            b0 = group_idx[0]

            # Step 1: Get golden solution text
            golden_text = self._get_golden_text(
                b0, questions, golden_answers, data_sources, golden_solutions
            )
            if golden_text is None:
                logger.warning(f"[StitchReward] No golden text for group uid={uid}, skipping")
                continue

            # Step 2: Split golden text into phases using the same splitter
            golden_phase_texts = self._split_golden(golden_text)
            if not golden_phase_texts:
                logger.warning(f"[StitchReward] Empty golden phases for uid={uid}, skipping")
                continue

            # Step 3: Run stitching for all responses in this group
            group_phase_texts = [phase_texts_batch[b] for b in group_idx]
            group_boundaries = [boundaries_batch[b] for b in group_idx]
            group_questions = [questions[b] for b in group_idx]
            group_answers = [golden_answers[b] for b in group_idx]
            group_sources = [data_sources[b] for b in group_idx]

            results = self.stitcher.stitch_group(
                questions=group_questions,
                response_phase_texts=group_phase_texts,
                golden_phase_texts=golden_phase_texts,
                golden_answers=group_answers,
                data_sources=group_sources,
                response_boundaries=group_boundaries,
                hf_model=self.hf_model,
                tokenizer=self.tokenizer,
            )

            # Step 4: Map SpliceResults back to batch indices
            for local_r, result in enumerate(results):
                b = group_idx[local_r]
                result.response_idx = b
                splice_results[b] = result

                if result.verified and result.splice_token_pos >= 0:
                    # Overwrite phase rewards: concentrate reward at splice phase
                    n_phases = len(boundaries_batch[b])
                    bounds = boundaries_batch[b]
                    splice_j = result.splice_j

                    for k in range(n_phases):
                        if k < splice_j:
                            phase_rewards[b, k] = self.pre_splice_adv
                        elif k == splice_j:
                            phase_rewards[b, k] = result.reward * self.splice_boost
                        else:
                            decay = self.post_splice_decay ** (k - splice_j)
                            phase_rewards[b, k] = result.reward * decay
                # else: keep the uniform incorrect_total / n_phases reward

        return phase_rewards, phase_mask_t, {"splice_results": splice_results}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_golden_text(
        self,
        b: int,
        questions: List[str],
        golden_answers: List[str],
        data_sources: List[str],
        golden_solutions: Optional[List[Optional[str]]],
    ) -> Optional[str]:
        """Return golden solution text for response b.

        Priority:
        1. If golden_gen provided → generate (answer-only dataset)
        2. Else use golden_solutions[b] (dataset with pre-written solutions)
        """
        if self.golden_gen is not None:
            results = self.golden_gen.generate_golden_paths(
                [questions[b]], [golden_answers[b]], [data_sources[b]]
            )
            return results[0]
        if golden_solutions is not None:
            return golden_solutions[b]
        return None

    def _split_golden(self, golden_text: str) -> List[str]:
        """Tokenize golden text, run splitter, return phase text strings.

        Uses HF model forward (if available) to compute log_probs/entropy
        for the splitter. Falls back to entropy ≈ 0 (uniform split) when
        hf_model is not provided.
        """
        if self.tokenizer is None:
            # Fallback: treat entire golden text as one phase
            return [golden_text]

        token_ids = self.tokenizer.encode(golden_text, add_special_tokens=False)
        if not token_ids:
            return [golden_text]

        golden_len = len(token_ids)
        ids_t = torch.tensor([token_ids], dtype=torch.long)  # (1, L)
        response_mask_golden = torch.ones(1, golden_len)      # all response tokens

        if self.hf_model is not None:
            device = next(self.hf_model.parameters()).device
            ids_t = ids_t.to(device)
            with torch.no_grad():
                logits = self.hf_model(input_ids=ids_t).logits.float()  # (1, L, V)
                lp = torch.log_softmax(logits, dim=-1)
                # log_prob[t] = log P(token t | token 0..t-1)
                # Shift: lp[:, :-1, :] predicts ids_t[:, 1:]
                log_probs_golden = torch.zeros(1, golden_len, device=device)
                for t in range(1, golden_len):
                    log_probs_golden[0, t] = lp[0, t - 1, token_ids[t]]
            # Approximate entropy ≈ -log_prob (rough but consistent with splitter)
            entropy_golden = -log_probs_golden
            response_mask_golden = response_mask_golden.to(device)
            ids_t_cpu = ids_t.cpu()
        else:
            entropy_golden = torch.zeros(1, golden_len)
            log_probs_golden = torch.zeros(1, golden_len)
            ids_t_cpu = ids_t

        boundaries_list = self.splitter.split(
            entropy=entropy_golden.cpu(),
            response_mask=response_mask_golden.cpu(),
            log_probs=log_probs_golden.cpu(),
            token_ids=ids_t_cpu,
            tokenizer=self.tokenizer,
        )
        boundaries = boundaries_list[0]  # single-element batch

        # Slice golden_text by token boundaries → phase texts
        phase_texts = []
        for k, start in enumerate(boundaries):
            end = boundaries[k + 1] if k + 1 < len(boundaries) else golden_len
            phase_tokens = token_ids[start:end]
            phase_texts.append(
                self.tokenizer.decode(phase_tokens, skip_special_tokens=True)
            )

        return phase_texts if phase_texts else [golden_text]
