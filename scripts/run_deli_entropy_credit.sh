#!/bin/bash
# =============================================================================
# DeliEntropySplitter + EntropyCredit Reward + PhaseAdvantage
#
# Phase boundaries: sliding-window entropy percentile + hard boundary at </think>
#   DeliEntropySplitter adds a forced split at the </think> token when present,
#   cleanly separating the reasoning trace from the output phase.
# Reward: cumulative entropy credit — sum(phase_rewards) = R_total per response
#         Phases with lower entropy than threshold get more credit
# Advantage: alpha * local + (1-alpha) * global
#
# Hardware: 1+ GPU (no extra GPU for reward — purely entropy-based)
#           Lightest configuration, fastest training step.
#           </think> boundary detection is purely tokenizer-level (no extra GPU).
#
# Usage:
#   bash scripts/run_deli_entropy_credit.sh
#   MODEL=Qwen3-14B NUM_GPUS=4 bash scripts/run_deli_entropy_credit.sh
#   bash scripts/run_deli_entropy_credit.sh algorithm.psi=0.90
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Hardware ---
MODEL=${MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
NUM_GPUS=${NUM_GPUS:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}

# --- Data ---
TRAIN_DATA=${TRAIN_DATA:-"data/processed/train/math.parquet"}
VAL_DATA=${VAL_DATA:-"data/processed/eval/math500.parquet"}

# --- Rollout ---
GROUP_SIZE=${GROUP_SIZE:-8}       # G: responses per prompt
BATCH_SIZE=${BATCH_SIZE:-16}      # prompts per step; total = BATCH_SIZE * GROUP_SIZE
EPOCHS=${EPOCHS:-3}

# --- Phase splitter (DeliEntropy) ---
# Same entropy parameters as PureEntropy, plus hard </think> boundary
ENTROPY_WINDOW_SIZE=${ENTROPY_WINDOW_SIZE:-10}
ENTROPY_PERCENTILE=${ENTROPY_PERCENTILE:-75.0}
PHASE_MIN_LEN=${PHASE_MIN_LEN:-10}
PHASE_MAX_K=${PHASE_MAX_K:-10}

# --- Reward: EntropyCredit ---
# sum(phase_rewards) = R_total for each response
PSI=${PSI:-0.95}                           # cumulative entropy decay: e_1 + e_2*psi + ...
DEFAULT_THRESHOLD_PCT=${DEFAULT_THRESHOLD_PCT:-90.0}
CORRECT_TOTAL=${CORRECT_TOTAL:-1.0}
INCORRECT_TOTAL=${INCORRECT_TOTAL:--1.0}
PARTIAL_TOTAL=${PARTIAL_TOTAL:-0.1}

# --- Advantage (PhaseAdvantage) ---
ALPHA=${ALPHA:-0.5}         # 0=pure global (GRPO), 1=pure local
DECAY_GAMMA=${DECAY_GAMMA:-0.0}   # 0=uniform token weight, >0=earlier tokens get more

# --- PPO ---
KL_COEF=${KL_COEF:-0.001}
CLIP_RATIO=${CLIP_RATIO:-0.2}

# --- Output ---
EXPERIMENT=${EXPERIMENT:-"deli-entropy-credit-$(basename $MODEL)-math"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

echo "============================================================"
echo " DeliEntropySplitter + EntropyCredit Reward"
echo " Model:           $MODEL"
echo " GPUs:            $NUM_GPUS   GPU mem util: $GPU_MEM_UTIL"
echo " Batch:           $BATCH_SIZE prompts x $GROUP_SIZE rollouts"
echo " Splitter:        window=$ENTROPY_WINDOW_SIZE  pct=$ENTROPY_PERCENTILE"
echo "                  min_len=$PHASE_MIN_LEN  max_K=$PHASE_MAX_K"
echo "                  [DELI] hard </think> boundary enabled"
echo " EntropyCredit:   psi=$PSI  thresh_pct=$DEFAULT_THRESHOLD_PCT"
echo "                  correct=$CORRECT_TOTAL  incorrect=$INCORRECT_TOTAL"
echo " Advantage:       alpha=$ALPHA  decay_gamma=$DECAY_GAMMA"
echo " Output:          $OUTPUT_DIR"
echo "============================================================"

python -m adpo.main \
    --config-name adpo_unified \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.clip_ratio="$CLIP_RATIO" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size="$BATCH_SIZE" \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
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
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=adpo \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    resources.num_gpus="$NUM_GPUS" \
    custom_reward_function.path=adpo/reward_functions.py \
    custom_reward_function.name=compute_score \
    "$@"
