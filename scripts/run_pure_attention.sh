#!/bin/bash
# =============================================================================
# PureEntropySplitter + AttentionReward + PhaseAdvantage
#
# Phase boundaries: sliding-window entropy percentile
# Reward: attention-flow propagation from last phase (exact-match) backward
#         Algorithm:
#           1. _partial_forward(model, input_ids, layer=L) → hidden_states (GPU)
#           2a. FAST PATH (OOM-safe, default): move hs to CPU immediately,
#               PCA-compress (L,D) → phase embeddings → cosine-sim confluence
#               Set: HIDDEN_PCA_COMPONENTS > 0  (e.g. 32)
#           2b. ATTENTION PATH: reconstruct Q·K^T at layer L on GPU,
#               build head-averaged (or PCA) phase attention matrix,
#               solve linear system A*r = r_last for phase rewards
#           3. solve_phase_rewards: A * r[0:m-2] = r[1:m-1], r_last from exact-match
# Advantage: alpha * local + (1-alpha) * global
#
# Hardware: 2 GPUs recommended
#   GPU 0: actor + rollout (vLLM) — set GPU_MEM_UTIL to leave room for HF model
#   GPU 1 (or GPU 0 with remaining VRAM): HF model for attention/hidden-state
#
# OOM tips:
#   - Set HIDDEN_PCA_COMPONENTS=32 (fast path, GPU freed after _partial_forward)
#   - Or keep ATTN_PCA_COMPONENTS=4 (PCA over heads, stays on attention path)
#   - Lower GPU_MEM_UTIL (e.g. 0.5) to leave VRAM for HF model
#
# Usage:
#   bash scripts/run_pure_attention.sh
#   HIDDEN_PCA_COMPONENTS=32 bash scripts/run_pure_attention.sh   # OOM-safe
#   NUM_GPUS=4 GPU_MEM_UTIL=0.6 bash scripts/run_pure_attention.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Hardware ---
MODEL=${MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
NUM_GPUS=${NUM_GPUS:-2}
# Leave VRAM headroom for the HF model loaded by AttentionReward
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}

# --- Data ---
TRAIN_DATA=${TRAIN_DATA:-"data/processed/train/math.parquet"}
VAL_DATA=${VAL_DATA:-"data/processed/eval/math500.parquet"}

# --- Rollout ---
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-3}

# --- Phase splitter (PureEntropy) ---
ENTROPY_WINDOW_SIZE=${ENTROPY_WINDOW_SIZE:-10}
ENTROPY_PERCENTILE=${ENTROPY_PERCENTILE:-75.0}
PHASE_MIN_LEN=${PHASE_MIN_LEN:-10}
PHASE_MAX_K=${PHASE_MAX_K:-10}

# --- Reward: AttentionReward ---
# Layer for hidden-state extraction (-1 = auto: floor(n_layers * 3/4))
ATTENTION_LAYER=${ATTENTION_LAYER:--1}
# Influence matrix A normalization: none | row | col | matrix
ATTENTION_NORM_MODE=${ATTENTION_NORM_MODE:-"row"}

# Outcome reward for last phase (from exact matching)
CORRECT_REWARD=${CORRECT_REWARD:-1.0}
INCORRECT_REWARD=${INCORRECT_REWARD:--1.0}
PARTIAL_REWARD=${PARTIAL_REWARD:-0.1}

# OOM-safe fast path: PCA on hidden states, skip attention reconstruction
# 0 = use attention reconstruction path (default)
# >0 = move hs to CPU immediately, cosine-sim between PCA phase embeddings
HIDDEN_PCA_COMPONENTS=${HIDDEN_PCA_COMPONENTS:-0}

# Attention-path PCA: reduce over heads instead of simple mean
# 0 = simple head mean (default); >0 = PCA over num_heads dimension
ATTN_PCA_COMPONENTS=${ATTN_PCA_COMPONENTS:-0}

# --- Advantage (PhaseAdvantage) ---
ALPHA=${ALPHA:-0.5}
DECAY_GAMMA=${DECAY_GAMMA:-0.0}

# --- PPO ---
KL_COEF=${KL_COEF:-0.001}
CLIP_RATIO=${CLIP_RATIO:-0.2}

# --- Output ---
EXPERIMENT=${EXPERIMENT:-"pure-attention-$(basename $MODEL)-math"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

echo "============================================================"
echo " PureEntropySplitter + AttentionReward"
echo " Model:           $MODEL"
echo " GPUs:            $NUM_GPUS   GPU mem util: $GPU_MEM_UTIL"
echo " Batch:           $BATCH_SIZE prompts x $GROUP_SIZE rollouts"
echo " Splitter:        window=$ENTROPY_WINDOW_SIZE  pct=$ENTROPY_PERCENTILE"
echo "                  min_len=$PHASE_MIN_LEN  max_K=$PHASE_MAX_K"
echo " AttentionReward: layer=$ATTENTION_LAYER  norm=$ATTENTION_NORM_MODE"
echo "                  correct=$CORRECT_REWARD  incorrect=$INCORRECT_REWARD"
if [ "$HIDDEN_PCA_COMPONENTS" -gt 0 ] 2>/dev/null; then
    echo "                  [FAST PATH] hidden_pca_components=$HIDDEN_PCA_COMPONENTS"
elif [ "$ATTN_PCA_COMPONENTS" -gt 0 ] 2>/dev/null; then
    echo "                  [HEAD PCA]  attn_pca_components=$ATTN_PCA_COMPONENTS"
else
    echo "                  [ATTN PATH] head mean confluence"
fi
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
    algorithm.pipeline_splitter=pure_entropy \
    algorithm.pipeline_reward=attention \
    algorithm.entropy_window_size="$ENTROPY_WINDOW_SIZE" \
    algorithm.entropy_percentile="$ENTROPY_PERCENTILE" \
    algorithm.phase_min_len="$PHASE_MIN_LEN" \
    algorithm.phase_max_K="$PHASE_MAX_K" \
    algorithm.attention_layer="$ATTENTION_LAYER" \
    algorithm.attention_norm_mode="$ATTENTION_NORM_MODE" \
    algorithm.correct_reward="$CORRECT_REWARD" \
    algorithm.incorrect_reward="$INCORRECT_REWARD" \
    algorithm.partial_reward="$PARTIAL_REWARD" \
    algorithm.attention_hidden_pca_components="$HIDDEN_PCA_COMPONENTS" \
    algorithm.attention_pca_components="$ATTN_PCA_COMPONENTS" \
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
