# InGPO: Information-Gated Policy Optimization

Chào mừng bạn đến với **InGPO** - một cải tiến của thuật toán **SPO (Segment Policy Optimization)** giúp tối ưu hóa việc huấn luyện mô hình ngôn ngữ lớn (LLM) trên các tác vụ suy luận.

## InGPO là gì?

InGPO thêm hai cơ chế thông minh vào quá trình tìm kiếm cây (tree search) của SPO:

1. **ValueShare (Chia sẻ)**: Khi phát hiện các đoạn văn bản dư thừa (có phân phối xác suất giống nhau), InGPO sẽ chia sẻ giá trị thay vì tính toán lại, giúp tiết kiệm thời gian.
2. **Prune (Cắt tỉa)**: Khi phát hiện các đoạn văn bản không liên quan đến thông tin cần thiết, InGPO sẽ loại bỏ chúng khỏi cây tìm kiếm, giảm chi phí tính toán.

Cả hai cơ chế này được kích hoạt dựa trên **Total Variation (TV) bound** - một ngưỡng thống kê đảm bảo tính chính xác.

## Cấu trúc thư mục

```
ingpo/
├── spo/                         # Code gốc của SPO (không sửa đổi)
├── ingpo_src/                   # Code mở rộng cho InGPO
│   ├── ingpo_main.py            # Điểm khởi chạy chính
│   └── ingpo_ext/               # Các thành phần cốt lõi
│       ├── core/                # Toán học: TV distance, thresholds, triggers
│       ├── inference_strategies/# Chiến lược xây dựng cây InGPO
│       └── episode_generators/  # Tạo episodes từ cây
├── configs/                     # File cấu hình cho các thí nghiệm
│   ├── ingpo_defaults.libsonnet # Cài đặt mặc định
│   ├── polIter_*.jsonnet        # Config cho từng model/dataset
│   ├── ablations/               # Config cho các ablation studies
│   └── baselines/               # Config cho các baseline (SPO, PPO, GRPO...)
├── scripts/                     # Script để chạy training, evaluation, analysis
└── tests/                       # Unit tests
```

## Bắt đầu nhanh

### Bước 1: Cài đặt môi trường

```bash
# Cài đặt dependencies và tải datasets
bash scripts/setup.sh
```

### Bước 2: Khởi động server vLLM (cần thiết để scoring)

Mở một terminal khác và chạy:

```bash
# Thay thế model path bằng đường dẫn local hoặc HuggingFace repo
bash scripts/start_vllm_server.sh /workspace/storage-shared/models/Qwen2.5-1.5B 8000 42 32 0

# Đặt biến môi trường cho API
export APP_OPENAI_VLLM_API_BASE=http://127.0.0.1:8000/v1
```

### Bước 3: Huấn luyện mô hình

```bash
# Chọn cấu hình cây (ví dụ: 666 = branch factor 6-6-6)
INGPO_TREE=666 bash scripts/train_ingpo_tree_MATH.sh
```

Một số script training có sẵn:
- `train_ingpo_tree_MATH.sh` - Huấn luyện trên dataset MATH
- `train_ingpo_tree_GSM8K.sh` - Huấn luyện trên dataset GSM8K
- `train_ingpo_tree_qwen1b_MATH.sh` - Dùng model Qwen 1.5B
- `train_debug.sh` - Chạy thử 2 iterations để kiểm tra

### Bước 4: Đánh giá mô hình

```bash
bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010
```

## Chạy các thí nghiệm

### Thí nghiệm chính từ paper

```bash
# Exp 1: Hiệu quả sample trên Qwen-MATH và Rho-GSM8K
TREES="444 666 888" MODELS="qwen rho" bash scripts/run_exp1_sample_efficiency.sh

# Exp 2: Tỷ lệ prune/share theo độ sâu cây
INGPO_TREE=666 bash scripts/run_exp2_prune_share_rate.sh

# Exp 3: So sánh overhead với SPO
INGPO_TREE=666 NUM_ITER=10 bash scripts/run_exp3_overhead.sh
```

### Ablation studies

```bash
# Chạy tất cả ablations
ABLATIONS="abl1 abl2 abl3 abl4 abl5 abl6 abl7" bash scripts/run_ablations.sh
```

| Tên run | Mục đích |
|---------|----------|
| `abl1-K{5,20}-m{50,200}` | Khảo sát K vs m (kích thước sample) |
| `abl2-eta-{0.005..0.05}` | Khảo sát ngưỡng η |
| `abl3-share-{parent,root}` | So sánh share target (parent vs root) |
| `abl4-{share-only,prune-only,neither}` | Tách riêng ảnh hưởng của Share và Prune |
| `abl5-y-per-dataset` | Precompute answer set Y một lần |
| `abl6-no-dkw` | Tắt DKW band → τ = η |
| `abl7-oracle-record` | Giữ lại PRUNE/SHARE edges để audit |

## Kiểm tra (Tests)

```bash
# Chạy unit tests
PYTHONPATH=ingpo_src python -m pytest tests/ -q

# Smoke test (kiểm tra config + tests, không cần GPU)
bash scripts/run_smoke.sh

# End-to-end debug run (2 iterations, cây độ sâu 2)
bash scripts/train_debug.sh
```

## Logging và Theo dõi

InGPO hỗ trợ logging **offline** (không cần internet):

### Metrics ghi lại tự động

- **Tỷ lệ tổng**: `share_rate`, `prune_rate`, số lượng expand/share/prune
- **Theo độ sâu**: metrics chi tiết cho mỗi tầng của cây
- **Answer set size**: Kích thước tập đáp án Y

### File demo (dễ đọc cho con người)

Hai file được ghi trong `<experiment_dir>/ingpo_demos/`:

1. **`demos.md`** - Định dạng Markdown, dễ đọc
2. **`demos.jsonl`** - JSON Lines, dùng cho phân tích

Xem demo trong khi training:
```bash
# Xem file Markdown (live update)
bash scripts/tail_demos.sh <tên_experiment>

# Xem file JSONL
bash scripts/tail_demos.sh <tên_experiment> jsonl
```

Phân tích sau khi chạy:
```bash
# Xem 20 demo đầu tiên
python scripts/inspect_demos.py experiments/<exp>/ingpo_demos/demos.jsonl

# Chỉ xem PRUNE demos ở depth 2
python scripts/inspect_demos.py experiments/<exp>/ingpo_demos/demos.jsonl \
    --action prune --depth 2 --limit 5

# Chỉ xem tổng kết (không hiển thị nội dung demo)
python scripts/inspect_demos.py experiments/<exp>/ingpo_demos/demos.jsonl --summary
```

## Cấu hình quan trọng

Các tham số mặc định nằm trong `configs/ingpo_defaults.libsonnet`:

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|------------------|---------|
| `K` | 10 | Số mẫu để tính AvgLP |
| `m` | 100 | Số mẫu để tính TV distance |
| `η` (eta) | 0.02 | Ngưỡng cho trigger |
| `demo_examples_per_tree` | 2 | Số demo SHARE/PRUNE lưu mỗi cây |

Bạn có thể override bất kỳ tham số nào trong file config `.jsonnet`:
```jsonnet
{
  ingpo+: {
    demo_examples_per_tree: 5,
    eta: 0.05,
  }
}
```

## Lưu ý kỹ thuật

- **vLLM scoring**: InGPO dùng endpoint `/v1/completions` với `echo=True, logprobs=1, max_tokens=0` để tính log-probability của các đáp án trong tập Y.
- **Không sửa code SPO**: Tất cả logic InGPO được thêm qua Python decorators và Jsonnet inheritance, code gốc của SPO giữ nguyên.
- **WandB optional**: Logging wandb tắt mặc định (`log_demos_to_wandb: false`), chỉ bật nếu bạn muốn upload lên wandb.ai.

## License

Giống như SPO - MIT License. Xem [`spo/LICENSE`](./spo/LICENSE) để biết chi tiết.

## Tài liệu tham khảo

- **[PLAN.md](./PLAN.md)**: Đặc tả chi tiết thuật toán InGPO
- **[SPO Repository](https://github.com/AIFrameResearch/SPO)**: Code gốc của SPO

---

## Notes: Các file thực hiện Build Tree

Dưới đây là danh sách các file chính chịu trách nhiệm xây dựng cây tìm kiếm (Tree Building) trong dự án:

### 1. Logic cốt lõi (Core Logic)
- **`ingpo/spo/tree_builder.py`** (hoặc `spo/tree_builder.py`): Chứa lớp `TreeBuilder` hoặc các hàm đệ quy `build_tree`, `expand_node`. Đây là nơi triển khai thuật toán DFS/BFS và quản lý cấu trúc cây.
- **`ingpo/spo/nodes.py`** (hoặc tương đương): Định nghĩa các lớp `Node`, `TreeNode`, lưu trữ trạng thái, log_prob, và value của từng node.

### 2. Scripts chạy thực nghiệm (Entry Points)
- **`ingpo/scripts/build_global_Y.py`**: Script chính để xây dựng cây toàn cục (global tree) cho tập dữ liệu.
- **`ingpo/scripts/run_tree_search.py`**: Script dùng để test và chạy tìm kiếm cây với các tham số cấu hình cụ thể.

### 3. Cấu hình (Configs)
- **`ingpo/configs/`** và **`ingpo/spo/configs/`**: Các file `.jsonnet` định nghĩa tham số như `max_depth`, `branching_factor`, `temperature`, và đường dẫn model.
  - Ví dụ: `polIter_qwen1b_ingpo_tree_MATH.jsonnet`

### 4. Tiện ích hỗ trợ (Utilities)
- **`ingpo/spo/utils/cache.py`**: Xử lý việc lưu/load cây đã build để tránh tính toán lại.
- **`ingpo/spo/utils/batch_inference.py`**: Các hàm wrapper để gọi model theo batch (cần kiểm tra xem đã tối ưu cho tree chưa).

> **Lưu ý hiệu năng:** Nếu cần tối ưu tốc độ, hãy tập trung vào `tree_builder.py` để chuyển đổi từ DFS sang BFS kết hợp Batch Inference Across Levels.
