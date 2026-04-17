#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL=${MODEL:-"/raid/models/models_rsync/thanhpv43/models/R1-Distill-1.5B"}
TRAIN_DATA=${TRAIN_DATA:-"/raid/models/models_rsync/thanhpv43/adpo/data/processed/train/math.parquet"} 
VAL_DATA=${VAL_DATA:-"/raid/models/models_rsync/thanhpv43/adpo/data/processed/eval/aime_2025.parquet"} 
NUM_GPUS=${NUM_GPUS:-2}
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-1}
JUDGE_TYPE=${JUDGE_TYPE:-"endpoint"}
PHASE_METHOD=${PHASE_METHOD:-"entropy"}
PHASE_PERCENTILE=${PHASE_PERCENTILE:-70.0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
EXPERIMENT=${EXPERIMENT:-"adpo-R1-Distill-1.5B-math-lora-v3"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}

max_token_len_per_gpu=10240

echo "============================================="
echo " ADPO Training — AoPS"
echo " Model:      $MODEL"
echo " GPUs:       $NUM_GPUS"
echo " Training data: $TRAIN_DATA"
echo " Evaluation data: $VAL_DATA"
echo "============================================="

python -m adpo.main_adpo \
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
    algorithm.judge_endpoint=http://10.254.138.189:8104 \
    algorithm.judge_model=Qwen3-14B \
    algorithm.judge_type="$JUDGE_TYPE" \
    algorithm.phase_method="$PHASE_METHOD" \
    algorithm.phase_percentile="$PHASE_PERCENTILE" \
    algorithm.use_kl_in_reward=False \
    "$@"
