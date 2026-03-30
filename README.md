# ADPO: Advantage Decomposition Policy Optimization

Phase-level credit assignment for GRPO-based LLM post-training on mathematical reasoning. Built on [verl](https://github.com/volcengine/verl).

## Core Idea

Standard GRPO computes a **response-level** advantage and broadcasts it uniformly to every token -- all reasoning steps receive the same signal regardless of quality.

**ADPO** decomposes the response into **phases** and scores each phase independently:

```
Response: [Phase 1: Setup] | [Phase 2: Key insight] | [Phase 3: Calculation] | [Phase 4: Answer]
                |                    |                        |                       |
           r_1 = 0.8            r_2 = 0.3                r_3 = 0.9              r_4 = 1.0
                |                    |                        |                       |
           A^(1) = +0.3         A^(2) = -0.7             A^(3) = +0.4           A^(4) = +0.5
```

### Three-Step Process

1. **Phase Boundary Detection**: Segment response using log-probability spikes. Tokens where `-log pi` is high indicate reasoning transitions (e.g., "Therefore", "Now consider", step numbers).

2. **LLM-as-Judge Scoring**: Each phase is evaluated independently by a judge model, producing per-phase rewards `r_k`.

3. **Phase-Level Advantages**: GRPO-style normalization applied at the phase level across G generations, then assigned back to tokens.

### Key Differences from Related Work

| Method | Auto boundaries? | Per-step reward? | No extra training? |
|---|---|---|---|
| GRPO | -- | No | Yes |
| PRM (Math-Shepherd) | No (human-defined) | Yes (learned RM) | No |
| OmegaPRM | No (MCTS) | Yes (learned RM) | No |
| **ADPO (this work)** | **Yes (log-prob)** | **Yes (LLM-as-Judge)** | **Yes** |

## Project Structure

```
adpo/
├── adpo/
│   ├── __init__.py
│   ├── adpo_algorithm.py      # Phase boundary detection, phase advantages, soft assignment
│   ├── adpo_trainer.py        # ADPOTrainer + monkey-patch for verl
│   ├── llm_judge.py           # LLM-as-Judge backends (vLLM, API, rule-based)
│   ├── reward_functions.py    # Math answer-matching reward functions
│   └── main_adpo.py           # Hydra entry point
├── configs/
│   └── adpo_trainer.yaml      # Default configuration
├── data/
│   └── prepare_datasets.py    # HuggingFace -> verl parquet conversion
├── evaluation/
│   └── evaluate.py            # vLLM-based evaluation
├── experiments/
│   └── v0_token_weight/       # Previous approach (token-level weighting)
├── scripts/
│   ├── prepare_all_data.sh
│   ├── train_math.sh          # + gsm8k, numina, openmathreas, aops, bigmathrl, finemath
│   ├── train_mixed.sh
│   ├── train_ablation_beta.sh # -> now train_ablation_phases.sh
│   ├── eval_all.sh
│   ├── eval_standard.sh
│   ├── eval_competition.sh
│   └── eval_passn.sh
├── docs/
│   └── HARDWARE_GUIDE.md
├── requirements.txt
├── setup.py
└── README.md
```

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ADPO Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. ROLLOUT          2. PHASE DECOMP         3. POLICY UPDATE       │
│  ┌────────────┐      ┌──────────────────┐    ┌──────────────┐      │
│  │ Generate G  │      │ Detect boundaries│    │ PPO clipped  │      │
│  │ responses   │─────>│ via -log pi      │───>│ objective    │      │
│  │ (vLLM)     │      │                  │    │ with A^(k)   │      │
│  └────────────┘      │ LLM-as-Judge     │    └──────────────┘      │
│                      │ scores each phase │                          │
│                      │                  │                           │
│                      │ Phase advantages  │                          │
│                      │ A_i^(k) per phase │                          │
│                      └──────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 0. Setup

```bash
conda create -n adpo python=3.10 -y && conda activate adpo
pip install verl[vllm]
pip install -e .
```

### 1. Prepare Data

```bash
bash scripts/prepare_all_data.sh
```

### 2. Train

```bash
# Rule-based judge (fast, uses answer matching for final phase)
bash scripts/train_math.sh

# LLM-as-Judge with local vLLM model (higher quality phase rewards)
bash scripts/train_math.sh algorithm.judge_type=vllm algorithm.judge_model=Qwen/Qwen2.5-7B-Instruct

# Customize phase detection
bash scripts/train_math.sh algorithm.phase_method=adaptive algorithm.phase_percentile=80.0

# In-phase advantage decreasing (earlier tokens get more credit)
bash scripts/train_math.sh algorithm.phase_decay_gamma=0.9

# Override model, GPUs, batch size
NUM_GPUS=4 BATCH_SIZE=64 MODEL=Qwen/Qwen2.5-Math-1.5B bash scripts/train_math.sh

# Ablation: sweep phase parameters
bash scripts/train_ablation_phases.sh
```

### 3. Evaluate

```bash
bash scripts/eval_all.sh --model_path checkpoints/adpo-qwen7b-math
bash scripts/eval_competition.sh --model_path checkpoints/adpo-qwen7b-math
bash scripts/eval_passn.sh --model_path checkpoints/adpo-qwen7b-math --n 8
```

## Configuration

### ADPO-Specific Parameters

| Parameter | Default | Description |
|---|---|---|
| `algorithm.phase_method` | adaptive | Boundary detection: "adaptive" or "threshold" |
| `algorithm.phase_percentile` | 85.0 | Percentile for adaptive boundary detection |
| `algorithm.phase_delta` | 2.0 | Fixed threshold for -log pi (if method=threshold) |
| `algorithm.phase_min_len` | 10 | Minimum tokens per phase |
| `algorithm.phase_max_K` | 10 | Maximum phases per response |
| `algorithm.phase_decay_gamma` | 0.0 | In-phase decay factor (0=off, >0=geometric decay) |
| `algorithm.judge_type` | rule | Judge backend: "rule", "vllm", or "api" |
| `algorithm.judge_model` | Qwen/Qwen2.5-7B-Instruct | Model for vllm/api judge |

### Judge Modes

| Mode | Speed | Quality | GPU Cost | When to use |
|---|---|---|---|---|
| `rule` | Fast | Low | None | Debugging, baselines |
| `vllm` | Medium | High | +1 GPU | Main experiments |
| `api` | Slow | Highest | API cost | Small-scale, best quality |

## Algorithm Details

### Phase Boundary Detection

The entropy profile `h_t = -log pi(a_t | s_t)` reveals reasoning transitions:

```
h_t: ░░▓░░░░░░░▓▓░░░░░░░░░░▓░░░░░░░░▓▓▓░░░
     ^         ^              ^         ^
     Phase 1   Phase 2        Phase 3   Phase 4
     (setup)   (key step)     (compute)  (answer)
```

**Adaptive method**: Use the p-th percentile of h_t as threshold, then select local maxima greedily.

### Soft Phase Assignment (Generalized Form)

Instead of hard boundaries, blend with Gaussian kernel:

```
w_{t,k} = exp(-(t - c_k)^2 / 2*sigma^2) / Z
A(t) = sum_k w_{t,k} * A^(k)
```

- sigma=0: hard assignment (default)
- sigma>0: smooth blending near boundaries
- sigma->inf: uniform (recovers GRPO)

### Theoretical Properties

1. **Unbiased**: Phase assignment is deterministic given the generated sequence, so the gradient estimator remains unbiased.
2. **Variance Reduction**: Per-phase rewards from LLM-as-Judge correlate more strongly with the causal contribution of each phase than a single outcome reward does.

## Datasets

### Training (7 datasets)

MATH, GSM8K, NuminaMath-1.5, OpenMathReasoning, AoPS-Instruct, Big-Math-RL-Verified, FineMath

### Evaluation (11 benchmarks)

MATH500, GSM8K test, AMC 2023, AIME 2024, AIME 2025, OlympiadBench, Minerva Math, Omni-MATH, HMMT, BRUMO, CMIMC

## Citation

```bibtex
@misc{adpo2026,
  title={Advantage Decomposition Policy Optimization: Phase-Level Credit Assignment for LLM Post-Training},
  year={2026}
}
```

## Acknowledgements

- [verl](https://github.com/volcengine/verl) -- Volcano Engine Reinforcement Learning for LLMs
- [GRPO](https://arxiv.org/abs/2402.03300) -- Group Relative Policy Optimization
- [Qwen2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B) -- Base model
