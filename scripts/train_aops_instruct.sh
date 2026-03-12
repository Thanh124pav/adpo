#!/bin/bash
# Train ADPO on aops_instruct
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-7B"}
TRAIN_DATA=${TRAIN_DATA:-"data/processed/train/aops_instruct.parquet"}
VAL_DATA=${VAL_DATA:-"data/processed/eval/math500.parquet"}
NUM_GPUS=${NUM_GPUS:-8}
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-3}
BETA=${BETA:-1.0}
EXPERIMENT=${EXPERIMENT:-"adpo-qwen7b-aops"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

echo "============================================="
echo " ADPO Training — aops_instruct"
echo " Model:      $MODEL"
echo " Beta:       $BETA"
echo " GPUs:       $NUM_GPUS"
echo "============================================="

python -m adpo.main_adpo     actor_rollout_ref.model.path="$MODEL"     actor_rollout_ref.rollout.n="$GROUP_SIZE"     actor_rollout_ref.rollout.tensor_model_parallel_size=1     actor_rollout_ref.actor.ppo_mini_batch_size=128     actor_rollout_ref.actor.ppo_epochs=1     actor_rollout_ref.actor.use_kl_loss=true     actor_rollout_ref.actor.kl_loss_coef=0.001     actor_rollout_ref.actor.clip_ratio=0.2     actor_rollout_ref.actor.loss_agg_mode=token-mean     algorithm.adpo_beta="$BETA"     algorithm.norm_adv_by_std_in_grpo=true     data.train_files="$TRAIN_DATA"     data.val_files="$VAL_DATA"     data.train_batch_size="$BATCH_SIZE"     data.max_prompt_length=1024     data.max_response_length=2048     trainer.total_epochs="$EPOCHS"     trainer.save_freq=50     trainer.test_freq=50     trainer.project_name=adpo     trainer.experiment_name="$EXPERIMENT"     trainer.default_local_dir="$OUTPUT_DIR"     resources.num_gpus="$NUM_GPUS"     custom_reward_function.path=adpo/reward_functions.py     custom_reward_function.name=compute_score     "$@"
