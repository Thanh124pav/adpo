# ADPO Hardware & Configuration Guide

Hướng dẫn chi tiết về cấu hình phần cứng và phần mềm để chạy ADPO ở các quy mô khác nhau.

---

## Mục lục

1. [Yêu cầu phần mềm](#1-yêu-cầu-phần-mềm)
2. [Cấu hình tối thiểu (Minimum)](#2-cấu-hình-tối-thiểu)
3. [Cấu hình khuyến nghị (Recommended)](#3-cấu-hình-khuyến-nghị)
4. [Cấu hình tốt nhất (Optimal)](#4-cấu-hình-tốt-nhất)
5. [Bảng tổng hợp](#5-bảng-tổng-hợp)
6. [Ước tính chi phí cloud](#6-ước-tính-chi-phí-cloud)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Yêu cầu phần mềm

| Component | Version | Ghi chú |
|---|---|---|
| Python | >= 3.10 | Khuyến nghị 3.10 hoặc 3.11 |
| CUDA | >= 12.1 | Bắt buộc cho GPU training |
| PyTorch | >= 2.1.0 | Cần build với CUDA support |
| vLLM | >= 0.4.0 | Rollout engine |
| verl | >= 0.2.0 | Training framework |
| Ray | >= 2.10.0 | Distributed orchestration |
| NCCL | >= 2.18 | GPU communication |
| OS | Ubuntu 20.04+ / CentOS 7+ | Linux bắt buộc cho training |

### Cài đặt nhanh

```bash
conda create -n adpo python=3.10 -y
conda activate adpo

# PyTorch with CUDA
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# verl + vLLM
pip install verl[vllm]

# Project
cd adpo
pip install -e .
```

---

## 2. Cấu hình tối thiểu

> Mục tiêu: Chạy được pipeline end-to-end, phục vụ debug và prototype.

### Hardware

| Component | Spec |
|---|---|
| **GPU** | 1x NVIDIA A100 40GB (hoặc 2x RTX 4090 24GB) |
| **CPU** | 8 cores (Intel Xeon hoặc AMD EPYC) |
| **RAM** | 64 GB |
| **Storage** | 200 GB SSD (cho model weights + datasets) |
| **Network** | Không yêu cầu đặc biệt (single node) |

### Model & Training Config

```bash
# Model nhỏ nhất có ý nghĩa
MODEL=Qwen/Qwen2.5-Math-1.5B

# Giảm tất cả batch sizes
NUM_GPUS=1
GROUP_SIZE=4          # G=4 thay vì 8
BATCH_SIZE=8          # 8 prompts/batch thay vì 128
EPOCHS=1

bash scripts/train_math.sh \
    actor_rollout_ref.rollout.max_new_tokens=1024 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    data.max_prompt_length=512 \
    data.max_response_length=1024
```

### VRAM Breakdown (1x A100 40GB, Qwen-1.5B)

```
Model weights (bf16):           ~3 GB
vLLM rollout KV cache:        ~10 GB
Reference model (offloaded):    ~0 GB (CPU offload)
Optimizer states (AdamW):       ~6 GB
Activations + gradients:        ~8 GB
Buffer:                         ~3 GB
─────────────────────────────────────
Total:                         ~30 GB  (fits in 40 GB)
```

### Giới hạn

- Training chậm (~5-10x so với cấu hình recommended)
- Group size G=4 cho advantage estimation kém chính xác hơn G=8
- Chỉ phù hợp cho dataset nhỏ (MATH ~12K, GSM8K ~7.5K)
- Không chạy được model 7B trên single 40GB GPU

### Cloud option rẻ nhất

| Provider | Instance | GPU | Giá/giờ |
|---|---|---|---|
| Lambda Labs | 1x A100 40GB | A100 40GB | ~$1.10/hr |
| RunPod | 1x A100 40GB | A100 40GB | ~$1.09/hr |
| Vast.ai | 1x RTX 4090 | RTX 4090 24GB | ~$0.30/hr |

---

## 3. Cấu hình khuyến nghị

> Mục tiêu: Training ổn định với model 7B, eval đầy đủ. Phù hợp cho nghiên cứu.

### Hardware

| Component | Spec |
|---|---|
| **GPU** | 4x NVIDIA A100 80GB |
| **CPU** | 32 cores |
| **RAM** | 256 GB |
| **Storage** | 1 TB NVMe SSD |
| **Network** | NVLink hoặc PCIe Gen4 x16 giữa các GPU |

### Model & Training Config

```bash
MODEL=Qwen/Qwen2.5-Math-7B

NUM_GPUS=4
GROUP_SIZE=8
BATCH_SIZE=64
EPOCHS=3

bash scripts/train_math.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false
```

### VRAM Breakdown (per GPU, 4x A100 80GB, Qwen-7B with FSDP)

```
Model weights (bf16, sharded):  ~3.5 GB  (14GB / 4 GPUs)
vLLM rollout KV cache:        ~15 GB
Reference model (sharded):     ~3.5 GB
Optimizer states (sharded):     ~7 GB
Activations + gradients:       ~20 GB
Gradient checkpointing saves:  ~10 GB
Buffer:                         ~5 GB
─────────────────────────────────────
Total per GPU:                 ~54 GB  (fits in 80 GB)
```

### Training Time Estimates (Qwen-7B, MATH 12.5K)

| Config | Time/Epoch | Total (3 epochs) |
|---|---|---|
| 4x A100 80GB, G=8, BS=64 | ~2.5 hours | ~7.5 hours |
| 4x A100 80GB, G=16, BS=32 | ~4 hours | ~12 hours |

### Evaluation Time (vLLM inference)

| Benchmark | Problems | Time (4x A100, greedy) |
|---|---|---|
| MATH500 | 500 | ~15 min |
| GSM8K test | 1319 | ~20 min |
| AIME 2024 | 30 | ~5 min |
| All 11 benchmarks | ~3000+ | ~1.5 hours |

### Cloud option

| Provider | Instance | GPU | Giá/giờ |
|---|---|---|---|
| Lambda Labs | 4x A100 80GB | 4x A100 | ~$4.40/hr |
| GCP | a2-ultragpu-4g | 4x A100 80GB | ~$5.00/hr |
| AWS | p4d.24xlarge (shared) | 4x A100 | ~$4.50/hr |

**Ước tính chi phí cho 1 experiment đầy đủ** (train 3 epochs + eval all):
- Training: ~7.5h x $4.40 = **~$33**
- Evaluation: ~1.5h x $4.40 = **~$7**
- **Tổng: ~$40/experiment**

---

## 4. Cấu hình tốt nhất

> Mục tiêu: Performance cao nhất, train model lớn, ablation study đầy đủ.

### Hardware

| Component | Spec |
|---|---|
| **GPU** | 8x NVIDIA H100 80GB (hoặc 8x A100 80GB) |
| **CPU** | 64+ cores |
| **RAM** | 512 GB+ |
| **Storage** | 2+ TB NVMe SSD (RAID 0 preferred) |
| **Network** | NVLink/NVSwitch 900 GB/s (H100) hoặc 600 GB/s (A100) |

### Model & Training Config

```bash
# Full-scale: Qwen-7B with optimal settings
MODEL=Qwen/Qwen2.5-Math-7B

NUM_GPUS=8
GROUP_SIZE=16          # Larger group = better advantage estimation
BATCH_SIZE=128         # Full batch
EPOCHS=3

bash scripts/train_math.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false
```

```bash
# Large model: Qwen-72B (cần tensor parallelism)
MODEL=Qwen/Qwen2.5-Math-72B

NUM_GPUS=8
GROUP_SIZE=8
BATCH_SIZE=32

bash scripts/train_math.sh \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    data.max_response_length=4096
```

### VRAM Breakdown (per GPU, 8x H100 80GB, Qwen-7B)

```
Model weights (bf16, sharded):  ~1.75 GB  (14GB / 8 GPUs)
vLLM rollout (TP=2):          ~12 GB
Reference model (sharded):     ~1.75 GB
Optimizer states (sharded):     ~3.5 GB
Activations + gradients:       ~15 GB
Buffer:                        ~10 GB
─────────────────────────────────────
Total per GPU:                 ~44 GB  (plenty of headroom)
```

### Training Time Estimates

| Model | GPUs | Time/Epoch (MATH) | Total (3 epochs) |
|---|---|---|---|
| Qwen-7B | 8x H100 | ~45 min | ~2.25 hours |
| Qwen-7B | 8x A100 | ~1.5 hours | ~4.5 hours |
| Qwen-72B | 8x H100 | ~6 hours | ~18 hours |

### Full Ablation Study Timeline (8x H100)

```
1. Data preparation                          ~30 min
2. Beta ablation (4 runs x 2 epochs each)    ~6 hours
3. Dataset ablation (7 datasets x 1 epoch)   ~10 hours
4. Best config: full training (3 epochs)     ~2.5 hours
5. Evaluation on all 11 benchmarks           ~45 min
6. pass@8 evaluation on key benchmarks       ~2 hours
─────────────────────────────────────────────────────
Total:                                       ~22 hours
```

### Cloud option

| Provider | Instance | GPU | Giá/giờ |
|---|---|---|---|
| Lambda Labs | 8x H100 SXM | 8x H100 | ~$19.60/hr |
| Lambda Labs | 8x A100 80GB | 8x A100 | ~$8.80/hr |
| GCP | a3-highgpu-8g | 8x H100 | ~$20.00/hr |
| AWS | p5.48xlarge | 8x H100 | ~$22.00/hr |

**Ước tính chi phí cho full ablation study**:
- 8x H100: ~22h x $19.60 = **~$430**
- 8x A100: ~35h x $8.80 = **~$308**

---

## 5. Bảng tổng hợp

| | Minimum | Recommended | Optimal |
|---|---|---|---|
| **GPU** | 1x A100 40GB | 4x A100 80GB | 8x H100 80GB |
| **CPU** | 8 cores | 32 cores | 64+ cores |
| **RAM** | 64 GB | 256 GB | 512 GB |
| **Storage** | 200 GB SSD | 1 TB NVMe | 2 TB NVMe |
| **Model** | Qwen-1.5B | Qwen-7B | Qwen-7B/72B |
| **Group size G** | 4 | 8 | 16 |
| **Batch size** | 8 | 64 | 128 |
| **Train time (MATH, 3ep)** | ~30 hours | ~7.5 hours | ~2.25 hours |
| **Cloud cost/experiment** | ~$33 | ~$40 | ~$50 |
| **Cloud cost (full study)** | N/A | ~$200 | ~$430 |

---

## 6. Ước tính chi phí cloud

### Kịch bản 1: Paper submission cơ bản

```
- 1 main experiment (7B, MATH, 3 epochs)       : ~$40
- 1 beta ablation (4 configs x 2 epochs)        : ~$60
- 3 eval runs (all benchmarks)                   : ~$20
                                          Total  : ~$120
```

### Kịch bản 2: Paper submission đầy đủ

```
- 3 main experiments (7B, 3 datasets, 3 epochs) : ~$120
- 1 beta ablation (4 configs x 2 epochs)         : ~$60
- 1 dataset ablation (7 datasets x 1 epoch)      : ~$100
- 10 eval runs                                    : ~$70
- Debugging & reruns (buffer 20%)                 : ~$70
                                          Total   : ~$420
```

### Kịch bản 3: Comprehensive study (Tier 2 venue)

```
- Model scale study (1.5B, 7B, 72B)              : ~$800
- Beta ablation (4 configs x 3 seeds)             : ~$360
- Dataset ablation (7 datasets)                   : ~$300
- Full eval suite (pass@1, pass@8, maj@8)         : ~$200
- Debugging & reruns (buffer 30%)                 : ~$500
                                          Total   : ~$2,160
```

---

## 7. Troubleshooting

### CUDA Out of Memory (OOM)

```bash
# Giải pháp 1: Giảm batch size
BATCH_SIZE=32 bash scripts/train_math.sh

# Giải pháp 2: Giảm response length
bash scripts/train_math.sh data.max_response_length=1024

# Giải pháp 3: Offload reference model sang CPU
bash scripts/train_math.sh actor_rollout_ref.ref.fsdp_config.param_offload=true

# Giải pháp 4: Giảm vLLM memory
bash scripts/train_math.sh actor_rollout_ref.rollout.gpu_memory_utilization=0.3

# Giải pháp 5: Dùng gradient checkpointing (default on)
bash scripts/train_math.sh actor_rollout_ref.model.enable_gradient_checkpointing=true
```

### Slow Training

```bash
# Tăng GPU utilization cho vLLM rollout
bash scripts/train_math.sh actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# Dùng tensor parallelism cho rollout (nếu multi-GPU)
bash scripts/train_math.sh actor_rollout_ref.rollout.tensor_model_parallel_size=2

# Giảm save/test frequency
bash scripts/train_math.sh trainer.save_freq=200 trainer.test_freq=200
```

### Multi-node Setup

```bash
# Node 0 (master)
ray start --head --port=6379

# Node 1, 2, ... (workers)
ray start --address=<master-ip>:6379

# Launch training (tự detect tất cả nodes)
NUM_GPUS=16 bash scripts/train_math.sh
```

### RTX 4090 (24GB) Workaround

RTX 4090 có VRAM nhỏ hơn A100 nhưng vẫn chạy được model 1.5B-7B:

```bash
# 2x RTX 4090: Qwen-1.5B
MODEL=Qwen/Qwen2.5-Math-1.5B NUM_GPUS=2 BATCH_SIZE=16 GROUP_SIZE=4 \
    bash scripts/train_math.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    data.max_response_length=1024

# 4x RTX 4090: Qwen-7B (tight fit)
MODEL=Qwen/Qwen2.5-Math-7B NUM_GPUS=4 BATCH_SIZE=8 GROUP_SIZE=4 \
    bash scripts/train_math.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    data.max_response_length=1024
```

---

## Checklist trước khi chạy

- [ ] CUDA version >= 12.1 (`nvidia-smi`)
- [ ] PyTorch detect GPU (`python -c "import torch; print(torch.cuda.device_count())"`)
- [ ] verl installed (`python -c "import verl; print(verl.__version__)"`)
- [ ] Data prepared (`ls data/processed/train/ data/processed/eval/`)
- [ ] Disk space sufficient (`df -h`)
- [ ] wandb login (optional: `wandb login`)
- [ ] Ray cluster running (multi-node: `ray status`)
