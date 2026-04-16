"""
Unified ADPO Trainer — composable PhaseSplitter → RewardComputer → AdvantageComputer pipeline.

Designed to be used as a drop-in replacement for any of the legacy trainers by
monkey-patching verl's compute_advantage at the Ray worker level.

Usage:
    trainer = ADPOTrainer(splitter, reward_computer, advantage_computer, config, tokenizer)
    trainer.patch_verl()

    # Inside verl compute loop:
    data = trainer.compute_advantages(data)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch field extraction helpers
# ---------------------------------------------------------------------------

def _extract_index(data) -> torch.Tensor:
    """Extract group UID index from verl DataProto."""
    batch = data.batch
    for key in ("uid", "index", "group_index", "prompt_id"):
        if key in batch:
            return batch[key]
    # Fallback: each response is its own group
    resp = batch.get("response_mask", batch.get("old_log_probs"))
    return torch.arange(resp.shape[0], device=resp.device)


def _extract_token_ids(data) -> Optional[torch.Tensor]:
    """Extract input_ids from verl DataProto if available."""
    batch = data.batch
    for key in ("input_ids", "token_ids"):
        if key in batch:
            return batch[key]
    return None


def _extract_questions(data, token_ids, response_mask, tokenizer) -> List[str]:
    """Decode question strings from the prompt portion of input_ids."""
    batch_size = response_mask.shape[0]
    questions = []
    for b in range(batch_size):
        q = ""
        if token_ids is not None and tokenizer is not None:
            active = response_mask[b].nonzero(as_tuple=True)[0]
            if len(active) > 0:
                resp_start = active[0].item()
                q = tokenizer.decode(token_ids[b, :resp_start], skip_special_tokens=True)
        questions.append(q)
    return questions


def _extract_full_responses(data, token_ids, response_mask, tokenizer) -> List[str]:
    """Decode full response strings."""
    batch_size = response_mask.shape[0]
    full_responses = []
    for b in range(batch_size):
        text = ""
        if token_ids is not None and tokenizer is not None:
            active = response_mask[b].nonzero(as_tuple=True)[0]
            if len(active) > 0:
                start, end = active[0].item(), active[-1].item() + 1
                text = tokenizer.decode(token_ids[b, start:end], skip_special_tokens=True)
        full_responses.append(text)
    return full_responses


def _extract_gt_and_source(data, batch_size: int):
    """Extract (golden_answers, data_sources) from verl DataProto."""
    golden_answers = [""] * batch_size
    data_sources = ["math"] * batch_size

    # Try non_tensor_batch first (verl standard)
    if hasattr(data, "non_tensor_batch"):
        ntb = data.non_tensor_batch
        if "reward_model" in ntb:
            rm_info = ntb["reward_model"]
            for b in range(batch_size):
                entry = rm_info[b] if hasattr(rm_info, "__getitem__") else rm_info
                if isinstance(entry, dict):
                    golden_answers[b] = entry.get("ground_truth", "")
        if "data_source" in ntb:
            ds_arr = ntb["data_source"]
            for b in range(batch_size):
                data_sources[b] = ds_arr[b] if hasattr(ds_arr, "__getitem__") else str(ds_arr)

    # Fallback: tensor batch fields
    batch = data.batch
    if hasattr(batch, "ground_truths") or "ground_truths" in batch:
        for b in range(batch_size):
            golden_answers[b] = batch["ground_truths"][b]
    if hasattr(batch, "data_sources") or "data_sources" in batch:
        for b in range(batch_size):
            data_sources[b] = batch["data_sources"][b]

    return golden_answers, data_sources


def _extract_golden_solutions(data, batch_size: int) -> Optional[List[Optional[str]]]:
    """Extract pre-written golden solutions if present (for StitchReward)."""
    if hasattr(data, "non_tensor_batch"):
        ntb = data.non_tensor_batch
        if "golden_solutions" in ntb:
            sols = ntb["golden_solutions"]
            return [sols[b] if hasattr(sols, "__getitem__") else sols for b in range(batch_size)]
    return None


def _compute_token_entropy(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy from log-probs using the identity H = -log p.

    This is an approximation (not the true categorical entropy over the vocab)
    but is consistent with how the splitters use entropy signal.

    Args:
        log_probs: (batch, seq_len) per-token log-prob of the chosen token.
        response_mask: (batch, seq_len) response token mask.

    Returns:
        entropy: (batch, seq_len) approx entropy, zero for non-response tokens.
    """
    entropy = -log_probs.float()
    entropy = entropy * response_mask
    return entropy


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class ADPOTrainer:
    """Composable trainer: PhaseSplitter → RewardComputer → AdvantageComputer.

    Args:
        splitter:           PhaseSplitter instance
        reward_computer:    BaseReward instance
        advantage_computer: PhaseAdvantage instance
        config:             OmegaConf / SimpleNamespace with algorithm sub-config
        tokenizer:          HuggingFace tokenizer (optional but recommended)
        hf_model:           HuggingFace model for HF-based splice scoring (optional)
    """

    def __init__(self, splitter, reward_computer, advantage_computer, config,
                 tokenizer=None, hf_model=None):
        self.splitter = splitter
        self.reward_computer = reward_computer
        self.advantage_computer = advantage_computer
        self.config = config
        self.tokenizer = tokenizer
        self.hf_model = hf_model

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def compute_advantages(self, data):
        """Run full ADPO pipeline on a verl DataProto batch.

        Flow:
            1. Extract batch fields (log_probs, response_mask, index, token_ids, …)
            2. Approximate entropy from log-probs
            3. Split each response into phases → boundaries
            4. Extract phase texts and outcome rewards
            5. Compute phase rewards (via reward_computer)
            6. Compute token-level advantages (via advantage_computer)
            7. Post-process stitching if StitchReward metadata present
            8. Write advantages back into data.batch["advantages"]

        Returns:
            data (DataProto) with data.batch["advantages"] filled in.
        """
        batch = data.batch
        log_probs = batch["old_log_probs"]          # (B, L)
        response_mask = batch["response_mask"]       # (B, L)
        batch_size, seq_len = response_mask.shape

        index = _extract_index(data)
        token_ids = _extract_token_ids(data)

        # 1. Entropy approximation
        entropy = _compute_token_entropy(log_probs, response_mask)

        # 2. Phase splitting
        boundaries_batch = self.splitter.split(
            entropy=entropy,
            response_mask=response_mask,
            log_probs=log_probs,
            token_ids=token_ids,
            tokenizer=self.tokenizer,
        )

        # 3. Extract text fields
        questions = _extract_questions(data, token_ids, response_mask, self.tokenizer)
        full_responses = _extract_full_responses(data, token_ids, response_mask, self.tokenizer)
        golden_answers, data_sources = _extract_gt_and_source(data, batch_size)
        golden_solutions = _extract_golden_solutions(data, batch_size)

        # 4. Extract phase texts via splitter
        phase_texts_batch: List[List[str]] = []
        for b in range(batch_size):
            active = response_mask[b].nonzero(as_tuple=True)[0]
            resp_end = active[-1].item() + 1 if len(active) > 0 else 0
            phases = self.splitter.extract_phase_texts(
                boundaries=boundaries_batch[b],
                response_length=resp_end,
                token_ids=token_ids[b] if token_ids is not None else None,
                tokenizer=self.tokenizer,
            )
            phase_texts_batch.append([p.text for p in phases])

        # 5. Outcome rewards (for reward computers that need them)
        outcome_rewards: List[float] = []
        for b in range(batch_size):
            if golden_answers[b]:
                r = compute_score(
                    data_source=data_sources[b],
                    solution_str=full_responses[b],
                    ground_truth=golden_answers[b],
                )
            else:
                r = 0.0
            outcome_rewards.append(float(r))

        # 6. Compute phase rewards
        phase_rewards, phase_mask, metadata = self.reward_computer.compute(
            boundaries_batch=boundaries_batch,
            response_mask=response_mask,
            index=index,
            entropy=entropy,
            outcome_rewards=outcome_rewards,
            phase_texts_batch=phase_texts_batch,
            questions=questions,
            golden_answers=golden_answers,
            data_sources=data_sources,
            full_responses=full_responses,
            golden_solutions=golden_solutions,
            hf_model=self.hf_model,
            tokenizer=self.tokenizer,
        )

        # 7. Compute token-level advantages
        token_advantages = self.advantage_computer.compute(
            phase_rewards=phase_rewards,
            phase_mask=phase_mask,
            response_mask=response_mask,
            boundaries_batch=boundaries_batch,
            index=index,
        )

        # 8. Post-process stitching if StitchReward produced splice_results
        splice_results = metadata.get("splice_results", {})
        if splice_results:
            from adpo.reward_computers.trajectory_stitching import compute_stitched_advantages
            algo = self.config.algorithm
            token_advantages = compute_stitched_advantages(
                token_advantages=token_advantages,
                response_mask=response_mask,
                stitch_results=splice_results,
                index=index,
                splice_boost=getattr(algo, "stitch_splice_boost", 2.0),
                pre_splice_advantage=getattr(algo, "stitch_pre_splice_adv", 0.0),
                post_splice_decay=getattr(algo, "stitch_post_splice_decay", 0.9),
            )

        data.batch["advantages"] = token_advantages
        return data

    # ------------------------------------------------------------------
    # verl integration
    # ------------------------------------------------------------------

    def patch_verl(self, tokenizer=None):
        """Monkey-patch verl's compute_advantage with this trainer's pipeline.

        Call this once per Ray worker after initialising the trainer.

        Args:
            tokenizer: override the trainer's tokenizer for this patch call.
        """
        import verl.trainer.ppo.ray_trainer as ray_trainer_module

        _trainer = self
        if tokenizer is not None:
            _trainer.tokenizer = tokenizer

        def _patched_compute_advantage(data):
            return _trainer.compute_advantages(data)

        ray_trainer_module.compute_advantage = _patched_compute_advantage
        logger.info("[ADPOTrainer] Patched verl compute_advantage successfully.")


# ---------------------------------------------------------------------------
# Ray-compatible task runner
# ---------------------------------------------------------------------------

def build_pipeline_from_config(config, tokenizer=None, hf_model=None):
    """Construct (splitter, reward_computer, advantage_computer) from config.

    Config keys (under config.algorithm):
        pipeline_splitter  : "pure_entropy" | "deli_entropy"   (default "pure_entropy")
        pipeline_reward    : "entropy" | "judge" | "attention" | "stitch"
                             (default "entropy")
        pipeline_advantage : "phase"  (currently only option)

    All algorithm sub-configs for each component are read by the component's
    __init__, so there's nothing more to wire here.
    """
    from adpo.phase_splitters.pure_entropy_splitter import PureEntropySplitter
    from adpo.phase_splitters.delimiter_entropy_splitter import DeliEntropySplitter
    from adpo.reward_computers.entropy_credit_reward import EntropyReward
    from adpo.reward_computers.judge_reward import JudgeReward
    from adpo.reward_computers.attention_reward import AttentionReward
    from adpo.reward_computers.stitcher_reward import StitchReward
    from adpo.advantages_computers.phase_advantage import PhaseAdvantage

    algo = config.algorithm
    splitter_name = getattr(algo, "pipeline_splitter", "pure_entropy")
    reward_name = getattr(algo, "pipeline_reward", "entropy")

    # --- Splitter ---
    if splitter_name == "deli_entropy":
        splitter = DeliEntropySplitter(config)
    else:
        splitter = PureEntropySplitter(config)

    # --- Reward computer ---
    if reward_name == "judge":
        reward_computer = JudgeReward(config)
    elif reward_name == "attention":
        reward_computer = AttentionReward(config)
    elif reward_name == "stitch":
        from adpo.reward_computers.trajectory_stitching import TrajectoryStitcher
        endpoint = getattr(algo, "stitch_endpoint", "")
        model = getattr(algo, "stitch_model", "")
        stitcher = TrajectoryStitcher(
            endpoint=endpoint,
            model=model,
            tokenizer=tokenizer,
            max_response_length=getattr(algo, "max_response_length", 2048),
            reward_decay=getattr(algo, "stitch_reward_decay", 0.1),
            max_golden_extensions=getattr(algo, "stitch_max_extensions", 5),
        )
        reward_computer = StitchReward(
            stitcher=stitcher,
            splitter=splitter,
            config=config,
            hf_model=hf_model,
            tokenizer=tokenizer,
        )
    else:  # default: entropy credit
        reward_computer = EntropyReward(config)

    # --- Advantage computer ---
    advantage_computer = PhaseAdvantage(config)

    return splitter, reward_computer, advantage_computer


try:
    import ray

    @ray.remote
    class UnifiedTaskRunner:
        """Ray remote actor that builds and runs the ADPO pipeline.

        Instantiate one per GPU worker. Automatically patches verl's
        compute_advantage so the standard training loop works unchanged.

        Args:
            config:    OmegaConf config (serialisable)
            tokenizer: HuggingFace tokenizer
            hf_model:  HuggingFace model (already on device, eval mode)
        """

        def __init__(self, config, tokenizer=None, hf_model=None):
            self.config = config
            self.tokenizer = tokenizer
            self.hf_model = hf_model
            self.trainer: Optional[ADPOTrainer] = None

        def setup(self):
            """Build the pipeline and patch verl. Call once after __init__."""
            splitter, reward_computer, advantage_computer = build_pipeline_from_config(
                self.config, self.tokenizer, self.hf_model
            )
            self.trainer = ADPOTrainer(
                splitter=splitter,
                reward_computer=reward_computer,
                advantage_computer=advantage_computer,
                config=self.config,
                tokenizer=self.tokenizer,
                hf_model=self.hf_model,
            )
            self.trainer.patch_verl(tokenizer=self.tokenizer)
            return True

        def compute_advantages(self, data):
            """Delegate to ADPOTrainer.compute_advantages."""
            if self.trainer is None:
                self.setup()
            return self.trainer.compute_advantages(data)

except ImportError:
    # Ray not installed — skip the remote actor definition
    pass
