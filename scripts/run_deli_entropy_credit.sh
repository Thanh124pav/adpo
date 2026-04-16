#!/bin/bash
# =============================================================================
# DeliEntropySplitter + EntropyCredit Reward + PhaseAdvantage
#
# Phase boundaries: sliding-window entropy percentile + hard boundary at </think>
#   The </think> token creates a forced split between reasoning and output phases,
#   giving the output phase its own dedicated reward slot.
# Reward: sum(phase_rewards) = R_total; low-entropy phases get more credit
# Advantage: alpha * local + (1-alpha) * global
#
# Hardware: works with 1+ GPU — no extra model for reward.
#
# Usage:
#   bash scripts/run_deli_entropy_credit.sh
#   MODEL=/path/to/model NUM_GPUS=4 bash scripts/run_deli_entropy_credit.sh
#   bash scripts/run_deli_entropy_credit.sh algorithm.psi=0.90
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
MODEL=${MODEL:-"/raid/models/models_rsync/thanhpv43/models/R1-Distill-1.5B"}
NUM_GPUS=${NUM_GPUS:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
max_token_len_per_gpu=5120

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TRAIN_DATA=${TRAIN_DATA:-"data/processed/train/math.parquet"}
VAL_DATA=${VAL_DATA:-"data/processed/eval/aime_2025.parquet"}

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
GROUP_SIZE=${GROUP_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-1}

# ---------------------------------------------------------------------------
# Phase splitter (DeliEntropy)
# Same entropy params as Pure; </think> boundary detected automatically via tokenizer.
# ---------------------------------------------------------------------------
ENTROPY_WINDOW_SIZE=${ENTROPY_WINDOW_SIZE:-10}
ENTROPY_PERCENTILE=${ENTROPY_PERCENTILE:-75.0}
PHASE_MIN_LEN=${PHASE_MIN_LEN:-10}
PHASE_MAX_K=${PHASE_MAX_K:-10}

# ---------------------------------------------------------------------------
# Reward: EntropyCredit
# ---------------------------------------------------------------------------
PSI=${PSI:-0.9}
DEFAULT_THRESHOLD_PCT=${DEFAULT_THRESHOLD_PCT:-90.0}
CORRECT_TOTAL=${CORRECT_TOTAL:-1.0}
INCORRECT_TOTAL=${INCORRECT_TOTAL:--1.0}
PARTIAL_TOTAL=${PARTIAL_TOTAL:-0.1}

# ---------------------------------------------------------------------------
# Advantage (PhaseAdvantage)
# ---------------------------------------------------------------------------
ALPHA=${ALPHA:-0.5}
DECAY_GAMMA=${DECAY_GAMMA:-0.95}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
EXPERIMENT=${EXPERIMENT:-"deli-entropy-credit-$(basename $MODEL)-math"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

echo "============================================================"
echo " DeliEntropySplitter + EntropyCredit Reward"
echo " Model:         $MODEL"
echo " GPUs:          $NUM_GPUS   GPU mem util: $GPU_MEM_UTIL"
echo " Batch:         $BATCH_SIZE prompts x $GROUP_SIZE rollouts"
echo " Splitter:      window=$ENTROPY_WINDOW_SIZE  pct=$ENTROPY_PERCENTILE"
echo "                min_len=$PHASE_MIN_LEN  max_K=$PHASE_MAX_K"
echo "                [DELI] hard </think> boundary enabled"
echo " EntropyCredit: psi=$PSI  thresh_pct=$DEFAULT_THRESHOLD_PCT"
echo "                correct=$CORRECT_TOTAL  incorrect=$INCORRECT_TOTAL"
echo " Advantage:     alpha=$ALPHA  decay_gamma=$DECAY_GAMMA"
echo " Output:        $OUTPUT_DIR"
echo "============================================================"

python -m adpo.main --config-name adpo_unified \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules='all-linear' \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.response_length=5120 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_prompt_length=384 \
    data.max_response_length=3840 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.shuffle=true \
    algorithm.pipeline_splitter=deli_entropy \
    algorithm.pipeline_reward=entropy \
    algorithm.entropy_window_size="$ENTROPY_WINDOW_SIZE" \
    algorithm.entropy_percentile="$ENTROPY_PERCENTILE" \
    algorithm.phase_min_len="$PHASE_MIN_LEN" \
    algorithm.phase_max_K="$PHASE_MAX_K" \
    algorithm.psi="$PSI" \
    algorithm.default_threshold_percentile="$DEFAULT_THRESHOLD_PCT" \
    algorithm.correct_total="$CORRECT_TOTAL" \
    algorithm.incorrect_total="$INCORRECT_TOTAL" \
    algorithm.partial_total="$PARTIAL_TOTAL" \
    algorithm.alpha="$ALPHA" \
    algorithm.decay_gamma="$DECAY_GAMMA" \
    trainer.use_legacy_worker_impl=disable \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=adpo \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.logger='["console","tensorboard","file"]' \
    trainer.val_before_train=True \
    trainer.validation_data_dir=outputs/val_debug \
    trainer.log_val_generations=10 \
    reward.custom_reward_function.path=adpo/reward_functions.py \
    reward.custom_reward_function.name=compute_score \
    "$@"
