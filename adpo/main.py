"""
Unified ADPO entry point.

Selects pipeline from config.algorithm.pipeline_reward and
config.algorithm.pipeline_splitter, then launches verl's PPO trainer
with the composable ADPOTrainer pipeline.

Usage:
    python -m adpo.main \\
        --config-name adpo_unified \\
        algorithm.pipeline_splitter=pure_entropy \\
        algorithm.pipeline_reward=entropy

    python -m adpo.main \\
        --config-name adpo_unified \\
        algorithm.pipeline_splitter=deli_entropy \\
        algorithm.pipeline_reward=attention \\
        algorithm.attention_hidden_pca_components=32

    python -m adpo.main \\
        --config-name adpo_unified \\
        algorithm.pipeline_splitter=pure_entropy \\
        algorithm.pipeline_reward=stitch \\
        algorithm.stitch_endpoint=http://localhost:8000 \\
        algorithm.stitch_rollout_mode=lite
"""

import logging

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from adpo.trainer import ADPOTrainer, build_pipeline_from_config

logger = logging.getLogger(__name__)


class UnifiedRayRunner(RayPPOTrainer):
    """Unified ADPO runner — extends RayPPOTrainer like legacy trainers.

    Builds the composable pipeline from config.algorithm.pipeline_* keys
    on first call to compute_advantage (lazy initialisation ensures the
    HF model is loaded in the correct Ray worker process / GPU context).

    Pipeline variants:
        pipeline_splitter: pure_entropy | deli_entropy
        pipeline_reward:   entropy | attention | judge | stitch
        pipeline_advantage: phase

    For attention and stitch: loads the training model (same checkpoint
    as actor) onto the GPU with the most free memory via _load_hf_model().
    Uses attention_hidden_pca_components > 0 to avoid CUDA OOM.
    """

    def __init__(self, config, tokenizer=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.tokenizer = tokenizer
        self._adpo_trainer = None   # lazy init on first compute_advantage call
        self._hf_model = None

    # ------------------------------------------------------------------
    # Lazy pipeline initialisation
    # ------------------------------------------------------------------

    def _needs_hf_model(self) -> bool:
        reward = getattr(self.config.algorithm, "pipeline_reward", "entropy")
        return reward in ("attention", "stitch")

    def _ensure_hf_model(self):
        """Load the HF model if this pipeline requires one.

        Uses the actor model path from config so the same checkpoint is
        used for both training and reward computation.
        Picks the GPU with the most free memory to avoid colliding with
        the actor / rollout.
        """
        if self._hf_model is not None:
            return
        if not self._needs_hf_model():
            return

        from adpo.reward_computers.attention_reward import _load_hf_model
        model_path = self.config.actor_rollout_ref.model.path
        logger.info(f"[UnifiedRayRunner] Loading HF model from {model_path}")
        self._hf_model = _load_hf_model(model_path)

    def _build_pipeline(self):
        self._ensure_hf_model()
        splitter, reward_computer, advantage_computer = build_pipeline_from_config(
            config=self.config,
            tokenizer=self.tokenizer,
            hf_model=self._hf_model,
        )
        self._adpo_trainer = ADPOTrainer(
            splitter=splitter,
            reward_computer=reward_computer,
            advantage_computer=advantage_computer,
            config=self.config,
            tokenizer=self.tokenizer,
            hf_model=self._hf_model,
        )
        pipeline_reward = getattr(self.config.algorithm, "pipeline_reward", "entropy")
        pipeline_splitter = getattr(self.config.algorithm, "pipeline_splitter", "pure_entropy")
        logger.info(
            f"[UnifiedRayRunner] Pipeline ready: "
            f"splitter={pipeline_splitter}, reward={pipeline_reward}"
        )
        print(
            f"[UnifiedRayRunner] Pipeline: splitter={pipeline_splitter}, "
            f"reward={pipeline_reward}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # verl interface
    # ------------------------------------------------------------------

    def compute_advantage(self, data):
        """Called by verl's RayPPOTrainer each rollout step."""
        if self._adpo_trainer is None:
            self._build_pipeline()
        return self._adpo_trainer.compute_advantages(data)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="adpo_unified", version_base=None)
def main(config: DictConfig):
    """Launch unified ADPO training."""
    config = migrate_legacy_reward_impl(config)
    print(OmegaConf.to_yaml(config))

    pipeline_reward = getattr(config.algorithm, "pipeline_reward", "entropy")
    pipeline_splitter = getattr(config.algorithm, "pipeline_splitter", "pure_entropy")

    print(
        f"\n[ADPO] Launching: splitter={pipeline_splitter}, reward={pipeline_reward}\n",
        flush=True,
    )

    # Attention and Stitch load an HF model inside the worker (explicit GPU
    # selection via _pick_best_gpu). Request only CPU from Ray to avoid
    # Ray reserving a GPU that the actor/rollout needs.
    task_runner_cls = ray.remote(num_cpus=1)(UnifiedRayRunner)
    run_ppo(config, task_runner_class=task_runner_cls)


if __name__ == "__main__":
    main()
