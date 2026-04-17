#!/bin/bash
# =============================================================================
# Train Pure-Entropy on math with DeepSeek-R1-Distill-Qwen3-1.5B (2 GPUs)
#
# Usage:
#   bash scripts/train_pure_entropy_math.sh
#
# Override any parameter via environment variables:
#   MODEL=path/to/model NUM_GPUS=4 BATCH_SIZE=32 bash scripts/train_pure_entropy_math.sh
#
# Or via Hydra CLI overrides:
#   bash scripts/train_pure_entropy_math.sh algorithm.entropy_percentile=80.0
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Configurable parameters (override via env vars) ---
MODEL=${MODEL:-"/raid/models/models_rsync/thanhpv43/models/Qwen3-0.6B"}
TRAIN_DATA=${TRAIN_DATA:-"data/processed/train/math.parquet"}
VAL_DATA=${VAL_DATA:-"data/processed/eval/aime_2025.parquet"}
NUM_GPUS=${NUM_GPUS:-1}
GROUP_SIZE=${GROUP_SIZE:-4}           # G: number of rollouts per prompt
BATCH_SIZE=${BATCH_SIZE:-2}          # number of prompts per batch (total responses = BATCH_SIZE * GROUP_SIZE)
EPOCHS=${EPOCHS:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
max_token_len_per_gpu=5120

# Pure-entropy algorithm parameters
ENTROPY_WINDOW_SIZE=${ENTROPY_WINDOW_SIZE:-10}     # sliding window k
ENTROPY_PERCENTILE=${ENTROPY_PERCENTILE:-75.0}     # percentile threshold
PHASE_MIN_LEN=${PHASE_MIN_LEN:-10}                 # minimum tokens per phase
PHASE_MAX_K=${PHASE_MAX_K:-10}                      # maximum phases per response
ATTENTION_LAYER=${ATTENTION_LAYER:--1}              # layer L for attention (-1 = auto: 3/4 depth)
ATTENTION_NORM_MODE=${ATTENTION_NORM_MODE:-"matrix"}

# Reward parameters
CORRECT_REWARD=${CORRECT_REWARD:-1.0}
INCORRECT_REWARD=${INCORRECT_REWARD:-0.0}
PARTIAL_REWARD=${PARTIAL_REWARD:-0.1}

# PPO / Actor parameters
KL_COEF=${KL_COEF:-0.001}
CLIP_RATIO=${CLIP_RATIO:-0.2}
PPO_EPOCHS=${PPO_EPOCHS:-1}

# Output
EXPERIMENT=${EXPERIMENT:-"pure-entropy-deepseek-r1-1.5b-math"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

echo "============================================="
echo " Pure-Entropy Training"
echo " Model:           $MODEL"
echo " Train data:      $TRAIN_DATA"
echo " GPUs:            $NUM_GPUS"
echo " Batch size:      $BATCH_SIZE (x${GROUP_SIZE} rollouts)"
echo " Entropy window:  $ENTROPY_WINDOW_SIZE"
echo " Entropy pct:     $ENTROPY_PERCENTILE"
echo " Attention layer: $ATTENTION_LAYER"
echo " Attn norm mode: $ATTENTION_NORM_MODE"
echo " Output:          $OUTPUT_DIR"
echo "============================================="
python -m adpo.main_pure_entropy \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules='all-linear' \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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
    data.max_response_length=3840\
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.shuffle=true \
    trainer.use_legacy_worker_impl=disable \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=adpo \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.logger='["console","tensorboard", "file"]' \
    trainer.val_before_train=True \
    trainer.validation_data_dir=outputs/val_debug \
    trainer.log_val_generations=10 \
    reward.custom_reward_function.path=adpo/reward_functions.py \
    reward.custom_reward_function.name=compute_score \
    algorithm.entropy_window_size="$ENTROPY_WINDOW_SIZE" \
    algorithm.entropy_percentile="$ENTROPY_PERCENTILE" \
    algorithm.phase_min_len="$PHASE_MIN_LEN" \
    algorithm.phase_max_K="$PHASE_MAX_K" \
    algorithm.attention_layer="$ATTENTION_LAYER" \
    algorithm.attention_norm_mode="$ATTENTION_NORM_MODE" \
    algorithm.correct_reward="$CORRECT_REWARD" \
    algorithm.incorrect_reward="$INCORRECT_REWARD" \
    algorithm.partial_reward="$PARTIAL_REWARD" \
    algorithm.use_kl_in_reward=False \
    "$@"