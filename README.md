# ADPO: Advantage Decomposition Policy Optimization

A token-level advantage weighting method for GRPO-based LLM post-training on mathematical reasoning tasks. Built on [verl](https://github.com/volcengine/verl).

## Core Idea

Standard GRPO computes a **response-level** advantage and broadcasts it uniformly to every token. This creates a **credit assignment problem** -- every token receives the same learning signal regardless of importance.

**ADPO** decomposes the advantage using log-probability weighting:

```
w_t = (-log pi(a_t | s_t))^beta / sum_{t'} (-log pi(a_{t'} | s_{t'}))^beta
A_i(t) = w_t * A_i
```

- Uncertain tokens (-log pi high) receive **stronger** gradient signal
- Confident tokens (-log pi low) receive **weaker** signal
- beta=0 recovers standard GRPO; beta=1 is default ADPO; beta -> inf concentrates signal on the most uncertain token

### Theoretical Properties

1. **Unbiased**: Since sum(w_t) = 1, the gradient estimator remains unbiased
2. **Variance Reduction**: When w_t correlates positively with the true causal contribution of token t, variance is reduced
3. **Implicit Entropy Regularization**: The objective implicitly encourages exploration at uncertain positions (similar to SAC)

## Project Structure

```
adpo/
├── adpo/
│   ├── __init__.py
│   ├── adpo_algorithm.py      # Core: token weight computation & advantage decomposition
│   ├── adpo_trainer.py        # ADPOTrainer (extends verl RayPPOTrainer) + monkey-patch
│   ├── reward_functions.py    # Reward functions for all math datasets
│   └── main_adpo.py           # Hydra entry point for training
├── configs/
│   └── adpo_trainer.yaml      # Default training configuration
├── data/
│   └── prepare_datasets.py    # Download & convert datasets to verl parquet format
├── evaluation/
│   └── evaluate.py            # vLLM-based evaluation with accuracy metrics
├── scripts/
│   ├── prepare_all_data.sh    # Prepare all train + eval datasets
│   ├── train_math.sh          # Train on MATH
│   ├── train_gsm8k.sh         # Train on GSM8K
│   ├── train_numina_math.sh   # Train on NuminaMath-1.5
│   ├── train_open_math_reasoning.sh  # Train on OpenMathReasoning
│   ├── train_aops_instruct.sh # Train on AoPS-Instruct
│   ├── train_big_math_rl.sh   # Train on Big-Math-RL-Verified
│   ├── train_fine_math.sh     # Train on FineMath
│   ├── train_mixed.sh         # Train on MATH+GSM8K+Numina combined
│   ├── train_ablation_beta.sh # Ablation: sweep beta in {0, 0.5, 1, 2}
│   ├── eval_all.sh            # Evaluate on all 11 benchmarks
│   ├── eval_standard.sh       # Evaluate on standard benchmarks
│   ├── eval_competition.sh    # Evaluate on competition math
│   └── eval_passn.sh          # Evaluate with pass@N / maj@N
├── requirements.txt
├── setup.py
└── README.md
```

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ADPO Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA PREP         2. TRAINING              3. EVALUATION    │
│  ┌─────────────┐      ┌──────────────────┐     ┌────────────┐  │
│  │ HuggingFace │      │ verl GRPO Loop   │     │ vLLM       │  │
│  │ Datasets    │─────>│                  │────>│ Generation │  │
│  │ -> Parquet  │      │ Rollout (vLLM)   │     │ + Scoring  │  │
│  └─────────────┘      │ Reward (custom)  │     └────────────┘  │
│                       │ ┌──────────────┐ │                      │
│                       │ │ ADPO Advtg   │ │     Benchmarks:      │
│                       │ │ w_t * A_i    │ │     MATH500,GSM8K    │
│                       │ └──────────────┘ │     AMC,AIME,HMMT    │
│                       │ Policy Update    │     OlympiadBench     │
│                       └──────────────────┘     Omni-MATH, etc.  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 0. Environment Setup

```bash
conda create -n adpo python=3.10 -y
conda activate adpo

# Install verl and dependencies
pip install verl[vllm]
pip install -e .
```

### 1. Prepare Data

```bash
# Prepare all datasets (train + eval)
bash scripts/prepare_all_data.sh

# Or prepare individual datasets
python data/prepare_datasets.py --dataset math --split train --output_dir data/processed/
python data/prepare_datasets.py --dataset math500 --output_dir data/processed/
```

### 2. Train

```bash
# Train on MATH with default settings (8 GPUs, beta=1.0)
bash scripts/train_math.sh

# Train on GSM8K
bash scripts/train_gsm8k.sh

# Train on mixed data (MATH + GSM8K + NuminaMath)
bash scripts/train_mixed.sh

# Custom: change model, GPU count, beta
NUM_GPUS=4 BETA=0.5 MODEL=Qwen/Qwen2.5-Math-1.5B bash scripts/train_math.sh

# Ablation: sweep beta values
bash scripts/train_ablation_beta.sh
```

### 3. Evaluate

```bash
# Evaluate on all benchmarks
bash scripts/eval_all.sh --model_path checkpoints/adpo-qwen7b-math

# Evaluate on competition math only
bash scripts/eval_competition.sh --model_path checkpoints/adpo-qwen7b-math

# Evaluate with pass@8
bash scripts/eval_passn.sh --model_path checkpoints/adpo-qwen7b-math --n 8
```

## Configuration

All training parameters are configurable via Hydra overrides or environment variables.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `algorithm.adpo_beta` | 1.0 | Weighting temperature. 0=GRPO, 1=ADPO, >1=sharper |
| `algorithm.norm_adv_by_std_in_grpo` | true | Normalize by group std (GRPO) vs subtract mean (Dr.GRPO) |
| `actor_rollout_ref.rollout.n` | 8 | Group size G (responses per prompt) |
| `actor_rollout_ref.actor.clip_ratio` | 0.2 | PPO clipping ratio |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | KL penalty coefficient |
| `actor_rollout_ref.actor.loss_agg_mode` | token-mean | Token loss aggregation |
| `data.train_batch_size` | 128 | Number of prompts per batch |
| `data.max_response_length` | 2048 | Maximum generation length |

### Hydra Override Examples

```bash
# Change model to 1.5B for debugging
python -m adpo.main_adpo actor_rollout_ref.model.path=Qwen/Qwen2.5-Math-1.5B

# Use Dr.GRPO normalization
python -m adpo.main_adpo algorithm.norm_adv_by_std_in_grpo=false

# Larger group size
python -m adpo.main_adpo actor_rollout_ref.rollout.n=16 data.train_batch_size=64
```

## Datasets

### Training Datasets

| Dataset | Size | Source |
|---|---|---|
| MATH | ~12.5K | `hendrycks/competition_math` |
| GSM8K | ~7.5K | `openai/gsm8k` |
| NuminaMath-1.5 | ~800K | `AI-MO/NuminaMath-1.5` |
| OpenMathReasoning | ~500K+ | `nvidia/OpenMathReasoning` |
| AoPS-Instruct | varies | `qq8933/AoPS-Instruct` |
| Big-Math-RL-Verified | varies | `SynthLabsAI/Big-Math-RL-Verified` |
| FineMath | ~200K (capped) | `HuggingFaceTB/finemath` |

### Evaluation Benchmarks

| Benchmark | Type | Source |
|---|---|---|
| MATH500 | Standard | `HuggingFaceH4/MATH-500` |
| GSM8K (test) | Standard | `openai/gsm8k` |
| AMC 2023 | Competition | `AI-MO/amc-aime-dataset` |
| AIME 2024 | Competition | `AI-MO/amc-aime-dataset` |
| AIME 2025 | Competition | `opencompass/AIME2025` |
| OlympiadBench | Olympiad | `lmms-lab/OlympiadBench` |
| Minerva Math | Standard | `math-ai/minerva_math` |
| Omni-MATH | Advanced | `KbsdJames/Omni-MATH` |
| HMMT | Competition | `keirp/HMMT` |
| BRUMO | Competition | `Idavidrein/BRUMO` |
| CMIMC | Competition | `I-Manuella/CMIMC_math` |

## How ADPO Works

### Algorithm

Given a group of G responses to the same prompt, ADPO:

1. **Response-level advantage** (same as GRPO):
   ```
   A_i = (r_i - mean) / (std + eps)
   ```

2. **Token-level decomposition** (ADPO contribution):
   ```
   w_t = (-log pi(a_t | s_t))^beta / sum_t'(-log pi(a_t' | s_t'))^beta
   A_i(t) = w_t * A_i
   ```

3. **Policy update** (standard clipped objective):
   ```
   L = (1/G) * sum_i sum_t A_i(t) * min(rho_t, clip(rho_t))
   ```

### Integration with verl

```python
from adpo.adpo_trainer import patch_verl_grpo_with_adpo

# Patch verl GRPO with ADPO advantage decomposition
patch_verl_grpo_with_adpo(beta=1.0)

# Then run verl standard training pipeline
trainer = RayPPOTrainer(config)
trainer.fit()
```

## Hardware Requirements

| Setup | GPUs | Model | Batch Size |
|---|---|---|---|
| Minimal | 2x A100 80GB | Qwen2.5-Math-1.5B | 32 |
| Standard | 8x A100 80GB | Qwen2.5-Math-7B | 128 |
| Large | 16x A100/H100 | Qwen2.5-Math-72B | 64 |

## Citation

```bibtex
@misc{adpo2026,
  title={Advantage Decomposition Policy Optimization: Token-Level Credit Assignment for LLM Post-Training},
  year={2026}
}
```

## Acknowledgements

- [verl](https://github.com/volcengine/verl) -- Volcano Engine Reinforcement Learning for LLMs
- [GRPO](https://arxiv.org/abs/2402.03300) -- Group Relative Policy Optimization (DeepSeek-Math)
- [Qwen2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B) -- Base model
