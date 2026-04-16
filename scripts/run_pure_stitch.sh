#!/bin/bash
# =============================================================================
# PureEntropySplitter + StitchReward + PhaseAdvantage
#
# Phase boundaries: sliding-window entropy percentile
# Reward: trajectory stitching — all-wrong GRPO groups get a "gold rescue":
#         find best splice point (j,i) in prefix × golden_phases,
#         rollout from stitched prefix, assign phase rewards to all responses.
#
# Algorithm:
#   For each all-wrong group:
#     1. Find golden solution (from dataset OR from golden_path_endpoint)
#     2. Split golden solution into phases (same splitter as actor)
#     3. Score all splice points (j,i): log P(golden[i] | response[:j])
#        via HF training model (splice_scorer=hf) or endpoint (splice_scorer=endpoint)
#     4. Rollout from best splice prefix:
#        lite — single rollout with all golden phases from i appended
#        full — iteratively add one golden phase at a time until correct
#     5. Assign spliced rewards to original responses + splice bonus
#
# Hardware: 2 GPUs recommended
#   GPU 0: actor + rollout (vLLM)
#   GPU 1 (or remaining VRAM on GPU 0): HF model for splice scoring
#
#   STITCH_ENDPOINT must point to a running vLLM server for rollout generation.
#   If GOLDEN_PATH_ENABLED=true, GOLDEN_PATH_ENDPOINT (or STITCH_ENDPOINT)
#   is also used to generate golden solutions on-the-fly.
#
# Usage:
#   STITCH_ENDPOINT=http://localhost:8000 bash scripts/run_pure_stitch.sh
#   ROLLOUT_MODE=lite SPLICE_SCORER=hf bash scripts/run_pure_stitch.sh
#   NUM_GPUS=4 GPU_MEM_UTIL=0.6 bash scripts/run_pure_stitch.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Hardware ---
MODEL=${MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
NUM_GPUS=${NUM_GPUS:-2}
# Leave VRAM headroom for the HF model used by splice scoring
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

# --- Reward: StitchReward ---
# vLLM endpoint for rollout generation (REQUIRED)
STITCH_ENDPOINT=${STITCH_ENDPOINT:-""}
STITCH_MODEL=${STITCH_MODEL:-"$MODEL"}

# Rollout mode:
#   full — pi(• | g[i], a[:j]): add golden phases one by one until correct (more thorough)
#   lite — pi(• | g[i:], a[:j]): give all golden phases from i, single rollout (faster)
ROLLOUT_MODE=${ROLLOUT_MODE:-"full"}

# Splice-point scorer:
#   hf       — log P(golden_phase[i] | response[:j]) via HF training model (default, accurate)
#   endpoint — same but via vLLM echo endpoint (faster but uses stale snapshot)
SPLICE_SCORER=${SPLICE_SCORER:-"hf"}

# Advantage shaping at splice point
STITCH_SPLICE_BOOST=${STITCH_SPLICE_BOOST:-2.0}
STITCH_PRE_SPLICE_ADV=${STITCH_PRE_SPLICE_ADV:-0.0}
STITCH_POST_SPLICE_DECAY=${STITCH_POST_SPLICE_DECAY:-0.9}
STITCH_REWARD_DECAY=${STITCH_REWARD_DECAY:-0.1}
STITCH_MAX_EXTENSIONS=${STITCH_MAX_EXTENSIONS:-5}

# Demo file: append per-step logs of all-wrong groups + splice results ("" to disable)
STITCH_DEMO_LOG=${STITCH_DEMO_LOG:-"stitch_demo.txt"}

# --- Golden path generation ---
# If your dataset has pre-written solutions, set GOLDEN_PATH_ENABLED=false (default).
# Set true to generate solutions on-the-fly via GOLDEN_PATH_ENDPOINT.
GOLDEN_PATH_ENABLED=${GOLDEN_PATH_ENABLED:-"false"}
GOLDEN_PATH_ENDPOINT=${GOLDEN_PATH_ENDPOINT:-"$STITCH_ENDPOINT"}
GOLDEN_PATH_MODEL=${GOLDEN_PATH_MODEL:-"$STITCH_MODEL"}
GOLDEN_PATH_MAX_ATTEMPTS=${GOLDEN_PATH_MAX_ATTEMPTS:-3}

# --- Advantage (PhaseAdvantage) ---
ALPHA=${ALPHA:-0.5}
DECAY_GAMMA=${DECAY_GAMMA:-0.0}

# --- PPO ---
KL_COEF=${KL_COEF:-0.001}
CLIP_RATIO=${CLIP_RATIO:-0.2}

# --- Output ---
EXPERIMENT=${EXPERIMENT:-"pure-stitch-$(basename $MODEL)-math"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/$EXPERIMENT"}

echo "============================================================"
echo " PureEntropySplitter + StitchReward"
echo " Model:           $MODEL"
echo " GPUs:            $NUM_GPUS   GPU mem util: $GPU_MEM_UTIL"
echo " Batch:           $BATCH_SIZE prompts x $GROUP_SIZE rollouts"
echo " Splitter:        window=$ENTROPY_WINDOW_SIZE  pct=$ENTROPY_PERCENTILE"
echo "                  min_len=$PHASE_MIN_LEN  max_K=$PHASE_MAX_K"
echo " StitchReward:    mode=$ROLLOUT_MODE  scorer=$SPLICE_SCORER"
echo "                  endpoint=$STITCH_ENDPOINT"
echo "                  splice_boost=$STITCH_SPLICE_BOOST"
echo "                  reward_decay=$STITCH_REWARD_DECAY  max_ext=$STITCH_MAX_EXTENSIONS"
if [ -n "$STITCH_DEMO_LOG" ]; then
    echo "                  demo_log=$STITCH_DEMO_LOG"
fi
echo " GoldenPath:      enabled=$GOLDEN_PATH_ENABLED"
echo " Advantage:       alpha=$ALPHA  decay_gamma=$DECAY_GAMMA"
echo " Output:          $OUTPUT_DIR"
echo "============================================================"

if [ -z "$STITCH_ENDPOINT" ]; then
    echo "WARNING: STITCH_ENDPOINT is not set. Rollout generation will fail."
    echo "         Set STITCH_ENDPOINT=http://localhost:8000 (or your vLLM server)."
fi

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
    algorithm.pipeline_reward=stitch \
    algorithm.entropy_window_size="$ENTROPY_WINDOW_SIZE" \
    algorithm.entropy_percentile="$ENTROPY_PERCENTILE" \
    algorithm.phase_min_len="$PHASE_MIN_LEN" \
    algorithm.phase_max_K="$PHASE_MAX_K" \
    algorithm.stitch_endpoint="$STITCH_ENDPOINT" \
    algorithm.stitch_model="$STITCH_MODEL" \
    algorithm.stitch_rollout_mode="$ROLLOUT_MODE" \
    algorithm.stitch_splice_scorer="$SPLICE_SCORER" \
    algorithm.stitch_splice_boost="$STITCH_SPLICE_BOOST" \
    algorithm.stitch_pre_splice_adv="$STITCH_PRE_SPLICE_ADV" \
    algorithm.stitch_post_splice_decay="$STITCH_POST_SPLICE_DECAY" \
    algorithm.stitch_reward_decay="$STITCH_REWARD_DECAY" \
    algorithm.stitch_max_extensions="$STITCH_MAX_EXTENSIONS" \
    algorithm.stitch_demo_log_path="$STITCH_DEMO_LOG" \
    algorithm.golden_path_enabled="$GOLDEN_PATH_ENABLED" \
    algorithm.golden_path_endpoint="$GOLDEN_PATH_ENDPOINT" \
    algorithm.golden_path_model="$GOLDEN_PATH_MODEL" \
    algorithm.golden_path_max_attempts="$GOLDEN_PATH_MAX_ATTEMPTS" \
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
