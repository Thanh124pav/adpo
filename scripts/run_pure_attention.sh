#!/bin/bash
# =============================================================================
# PureEntropySplitter + AttentionReward + PhaseAdvantage
#
# Phase boundaries: sliding-window entropy percentile
# Reward: attention-flow propagation from last phase (exact-match) backward
#
# Algorithm (3 paths — choose via env vars):
#   DEFAULT (HIDDEN_PCA_COMPONENTS=0, USE_DIRECT=false):
#     1. _partial_forward(model, input_ids, layer=L) → hidden_states
#     2. Reconstruct Q·K^T at layer L → head-averaged phase attention matrix
#     3. Solve A * r[0:m-2] = r[1:m-1], r_last from exact-match
#
#   FAST PATH (HIDDEN_PCA_COMPONENTS > 0, e.g. 32):
#     1. _partial_forward → hidden_states  (GPU freed immediately after)
#     2. PCA on hidden states (CPU) → phase embeddings → cosine-sim confluence
#     3. Solve linear system as above
#     Avoids Q·K^T on GPU entirely — best for OOM prevention.
#
#   DIRECT PATH (USE_DIRECT_ATTENTION=true):
#     1. Full forward pass with output_attentions=True (requires eager attn)
#     2. Read outputs.attentions[layer_L] directly (no reconstruction)
#     3. Slice to response tokens, build phase attention, solve
#
# Hardware: 2 GPUs recommended
#   GPU 0: actor + rollout (vLLM)    — GPU_MEM_UTIL controls vLLM's share
#   GPU 1 (or remaining VRAM):        HF model for attention/hidden-state
#   AttentionReward auto-picks the GPU with most free VRAM for the HF model.
#
# OOM tips:
#   HIDDEN_PCA_COMPONENTS=32    → fast path, no Q·K^T on GPU
#   ATTN_PCA_COMPONENTS=4       → PCA over heads (attention path, less memory)
#   GPU_MEM_UTIL=0.35            → more VRAM left for HF model
#
# Usage:
#   bash scripts/run_pure_attention.sh
#   HIDDEN_PCA_COMPONENTS=32 bash scripts/run_pure_attention.sh
#   USE_DIRECT_ATTENTION=true  bash scripts/run_pure_attention.sh
#   NUM_GPUS=4 GPU_MEM_UTIL=0.35 bash scripts/run_pure_attention.sh
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
# Leave VRAM headroom for the HF model (AttentionReward loads it separately)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
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
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-1}

# ---------------------------------------------------------------------------
# Phase splitter (PureEntropy)
# ---------------------------------------------------------------------------
ENTROPY_WINDOW_SIZE=${ENTROPY_WINDOW_SIZE:-10}
ENTROPY_PERCENTILE=${ENTROPY_PERCENTILE:-75.0}
PHASE_MIN_LEN=${PHASE_MIN_LEN:-10}
PHASE_MAX_K=${PHASE_MAX_K:-10}

# ---------------------------------------------------------------------------
# Reward: AttentionReward
# ---------------------------------------------------------------------------
# Layer for hidden-state / attention extraction (-1 = auto: floor(n_layers*3/4))
ATTENTION_LAYER=${ATTENTION_LAYER:--1}
# Influence matrix A normalisation: none | row | col | matrix
ATTENTION_NORM_MODE=${ATTENTION_NORM_MODE:-"matrix"}

# Outcome reward for last phase (exact-match)
CORRECT_REWARD=${CORRECT_REWARD:-1.0}
INCORRECT_REWARD=${INCORRECT_REWARD:-0.0}
PARTIAL_REWARD=${PARTIAL_REWARD:-0.1}

# Solve mode:
#   form b (default): r[0] = FIRST_PHASE_REWARD (fixed), r_last as additive bias at last phase
#   form a (legacy):  FIRST_PHASE_REWARD="" → r[last] fixed from outcome, solve backward
FIRST_PHASE_REWARD=${FIRST_PHASE_REWARD:-0.5}

# --- Path selection ---
# Direct attention (requires eager attn impl, reads outputs.attentions directly)
USE_DIRECT_ATTENTION=${USE_DIRECT_ATTENTION:-false}

# Hidden-state PCA fast path (skips Q·K^T entirely; GPU freed after _partial_forward)
# 0 = use attention reconstruction (default); >0 = PCA with n components (e.g. 32)
HIDDEN_PCA_COMPONENTS=${HIDDEN_PCA_COMPONENTS:-0}

# Attention-path PCA: reduce over heads (0 = simple head mean; >0 = PCA n heads)
ATTN_PCA_COMPONENTS=${ATTN_PCA_COMPONENTS:-0}

# ---------------------------------------------------------------------------
# Advantage (PhaseAdvantage)
# ---------------------------------------------------------------------------
ALPHA=${ALPHA:-0.5}
DECAY_GAMMA=${DECAY_GAMMA:-0.95}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
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
if [ "$USE_DIRECT_ATTENTION" = "true" ]; then
    echo "                  [DIRECT PATH] output_attentions=True (eager only)"
elif [ "${HIDDEN_PCA_COMPONENTS:-0}" -gt 0 ] 2>/dev/null; then
    echo "                  [FAST PATH]   hidden_pca_components=$HIDDEN_PCA_COMPONENTS"
elif [ "${ATTN_PCA_COMPONENTS:-0}" -gt 0 ] 2>/dev/null; then
    echo "                  [HEAD PCA]    attn_pca_components=$ATTN_PCA_COMPONENTS"
else
    echo "                  [RECON PATH]  hidden states → Q·K^T reconstruction"
fi
echo " Advantage:       alpha=$ALPHA  decay_gamma=$DECAY_GAMMA  mode=$PIPELINE_ADVANTAGE"
echo " Output:          $OUTPUT_DIR"
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
    algorithm.pipeline_splitter=pure_entropy \
    algorithm.pipeline_reward=attention \
    algorithm.entropy_window_size="$ENTROPY_WINDOW_SIZE" \
    algorithm.entropy_percentile="$ENTROPY_PERCENTILE" \
    algorithm.phase_min_len="$PHASE_MIN_LEN" \
    algorithm.phase_max_K="$PHASE_MAX_K" \
    algorithm.attention_layer="$ATTENTION_LAYER" \
    algorithm.attention_norm_mode="$ATTENTION_NORM_MODE" \
    algorithm.attention_use_direct="$USE_DIRECT_ATTENTION" \
    algorithm.attention_fixed_first_reward="$FIRST_PHASE_REWARD" \
    algorithm.correct_reward="$CORRECT_REWARD" \
    algorithm.incorrect_reward="$INCORRECT_REWARD" \
    algorithm.partial_reward="$PARTIAL_REWARD" \
    algorithm.attention_hidden_pca_components="$HIDDEN_PCA_COMPONENTS" \
    algorithm.attention_pca_components="$ATTN_PCA_COMPONENTS" \
    algorithm.alpha="$ALPHA" \
    algorithm.decay_gamma="$DECAY_GAMMA" \
    algorithm.pipeline_advantage="$PIPELINE_ADVANTAGE" \
    algorithm.hybrid_correct_threshold="$HYBRID_CORRECT_THRESHOLD" \
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
