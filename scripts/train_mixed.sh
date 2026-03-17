#!/bin/bash
# Train ADPO on MIXED dataset (MATH + GSM8K + NuminaMath combined)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL=${MODEL:-"Qwen3-4B"}
NUM_GPUS=${NUM_GPUS:-1}
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
EXPERIMENT=${EXPERIMENT:-"adpo-qwen7b-mixed"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

MERGED_DATA="data/processed/train/mixed.parquet"
if [ ! -f "$MERGED_DATA" ]; then
    echo ">>> Merging training datasets..."
    python -c "
import pandas as pd, glob
dfs = []
for f in glob.glob('data/processed/train/*.parquet'):
    name = f.split('/')[-1].replace('.parquet', '')
    if name in ['math', 'gsm8k', 'numina_math']:
        df = pd.read_parquet(f)
        dfs.append(df)
        print(f'  Loaded {name}: {len(df)} examples')
merged = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
merged.to_parquet('$MERGED_DATA', index=False)
print(f'Merged: {len(merged)} examples')
"
fi

echo "============================================="
echo " ADPO Training — Mixed (MATH+GSM8K+Numina)"
echo " Model: $MODEL"
echo "============================================="

python -m adpo.main_adpo \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +algorithm.norm_adv_by_std_in_grpo=true \
    data.train_files="$MERGED_DATA" \
    data.val_files=data/processed/eval/math500.parquet \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=adpo \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    resources.num_gpus="$NUM_GPUS" \
    custom_reward_function.path=adpo/reward_functions.py \
    custom_reward_function.name=compute_score \
    "$@"
