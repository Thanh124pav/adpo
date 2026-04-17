# Chỉ Mục Công Việc — Tái Cấu Trúc ADPO

Nhánh làm việc: `reconstruct-project`

---

## PHASE A — Sửa Lỗi Trong Code Hiện Có (Ưu tiên cao nhất) [DONE]

| # | File | Lỗi | Cách sửa |
|---|------|-----|---------|
| A1 | Tất cả class | `super(self, ClassName).__init__()` | Đổi thành `super().__init__()` |
| A2 | `reward_computers/base.py` | `compute()` thiếu `self` | Thêm `self` làm tham số đầu tiên |
| A3 | `reward_computers/attention_reward.py` | `batch_size` dùng trước khi khai báo | Chuyển `batch_size, seq_len = response_mask.shape` lên đầu hàm |
| A4 | `reward_computers/attention_reward.py` | Thiếu `continue` sau `n_phases <= 1` | Thêm `continue` sau dòng `phase_rewards_batch.append(...)` |
| A5 | `reward_computers/attention_reward.py` | `from adpo.pure_entropy_algorithm import _partial_forward` | Xóa import này — `_partial_forward` đã định nghĩa trong file |
| A6 | `reward_computers/attention_reward.py` | Trả về `List[np.ndarray]` thay vì `Tuple[Tensor, Tensor, dict]` | Xem hướng dẫn B2 |
| A7 | `reward_computers/judge_reward.py` | `self.phase_method` không tồn tại | Xóa dòng logger.info đó |
| A8 | `reward_computers/judge_reward.py` | Trả về `phase_rewards` thiếu mask và dict | `return phase_rewards, phase_mask_tensor, {}` |
| A9 | `phase_splitters/pure_entropy_splitter.py` | `split()` thiếu `log_probs=None` | Thêm tham số vào signature cho khớp base |

---

## PHASE B — Hoàn Thiện Các File Đang Dở

### B1. `reward_computers/entropy_credit_reward.py` — Hoàn thành bị truncate

**Tham số `__init__`:**
```
config.algorithm.psi               → self.psi        (default 0.95)
config.algorithm.default_threshold_percentile → self.threshold_pct (default 90.0)
config.algorithm.correct_total     → self.correct_total  (default 1.0)
config.algorithm.incorrect_total   → self.incorrect_total (default -1.0)
config.algorithm.partial_total     → self.partial_total  (default 0.1)
```

**`compute()` signature:**
```python
def compute(self, boundaries_batch, response_mask, index,
            entropy,           # Tensor (batch, seq_len) — BẮT BUỘC
            outcome_rewards,   # List[float] — BẮT BUỘC
            **context)
-> Tuple[Tensor (batch, max_K), Tensor (batch, max_K), dict]
```

**Luồng bên trong** — lấy code từ file cũ:
1. Gọi `compute_phase_cumulative_entropy(entropy, response_mask, boundaries_batch, psi=self.psi)`
   → Code: `entropy_credit_algorithm.py:108-166`
2. Gọi `compute_entropy_credit_rewards(cum_entropy, outcome_rewards, index, correct_total=..., incorrect_total=..., partial_total=..., default_percentile=...)`
   → Code: `entropy_credit_algorithm.py:173-314`
3. Convert `List[np.ndarray]` → `Tensor (batch, max_K)` và tạo `phase_mask_tensor`
4. `return phase_rewards, phase_mask_tensor, {}`

**Lưu ý:** Cả 2 hàm helper đó có thể import từ `entropy_credit_algorithm.py` hoặc copy vào file mới.

---

### B2. Fix return type `AttentionReward.compute()`

Hiện tại trả về `List[np.ndarray]`. Cần convert sang tensor + tạo mask:

```python
# Sau vòng lặp for b in range(batch_size)...
max_K = max(len(r) for r in phase_rewards_batch)
device = response_mask.device
phase_rewards_tensor = torch.zeros(batch_size, max_K, device=device)
phase_mask_tensor = torch.zeros(batch_size, max_K, device=device)
for b, rewards in enumerate(phase_rewards_batch):
    k = len(rewards)
    phase_rewards_tensor[b, :k] = torch.tensor(rewards, dtype=torch.float32)
    phase_mask_tensor[b, :k] = 1.0
return phase_rewards_tensor, phase_mask_tensor, {}
```

---

## PHASE C — Implement File Còn Trống

### C1. `reward_computers/stitcher_reward.py`

**Constructor nhận 2 objects:**
```python
def __init__(self, stitcher: TrajectoryStitcher, splitter: PhaseSplitter, config)
```

**`compute()` signature:**
```python
def compute(self, boundaries_batch, response_mask, index,
            outcome_rewards,    # List[float]
            phase_texts_batch,  # List[List[str]]
            questions,          # List[str]
            golden_answers,     # List[str]
            data_sources,       # List[str]
            entropy=None,       # Tensor (batch, seq_len) — cho entropy splice scoring
            log_probs=None,     # Tensor (batch, seq_len) — nếu dùng DeliEntropySplitter
            **context)
-> Tuple[Tensor (batch, max_K), Tensor (batch, max_K), dict]
  # dict = {"splice_results": Dict[int, SpliceResult]}
```

**Luồng bên trong:**
1. Tìm all-wrong groups: `index` + `outcome_rewards` → `all_wrong: Dict[uid, List[int]]`
2. Với mỗi all-wrong group:
   a. Gọi `GoldenPathGenerator` để lấy `golden_text` (tái dùng logic `adpo_trainer.py:687-733`)
   b. Gọi endpoint `echo=True, logprobs=1` trên `golden_text` → lấy `token_logprobs`
   c. Xấp xỉ: `log_probs_golden[t] = token_logprobs[t]`, `entropy_golden[t] ≈ -log_probs_golden[t]`
   d. Tạo `response_mask_golden` toàn 1s với shape `(1, len_golden_tokens)`
   e. Gọi `self.splitter.split(entropy=entropy_golden, response_mask=response_mask_golden, log_probs=log_probs_golden)` → `golden_boundaries`
   f. Slice `golden_text` theo `golden_boundaries` → `golden_phase_texts: List[str]`
   g. Gọi `self.stitcher.stitch_group(questions, phase_texts_batch_group, golden_phase_texts, golden_answers, data_sources, boundaries_group)`
3. Convert `SpliceResult` → `phase_rewards`:
   - Token trước splice: reward = 0.0
   - Token tại splice: reward = `result.reward`
   - Token sau splice: reward giảm dần
4. `return phase_rewards, phase_mask_tensor, {"splice_results": splice_results_dict}`

**Sửa `TrajectoryStitcher.stitch_group()` (`trajectory_stitching.py:466`):**
- Xóa tham số `golden_path: str`
- Thêm tham số `golden_phase_texts: List[str]`  — phases đã chia sẵn từ bên ngoài
- Xóa dòng gọi `self.segment_golden_path(...)` (dòng 491-493)
- Dùng `golden_phase_texts` trực tiếp thay vào chỗ `golden_phases`
- Xóa method `segment_golden_path()` (dòng 184-260) sau khi đã migrate

---

### C2. `advantages_computers/phase_advantage.py`

**Class `PhaseAdvantageComputer`:**
```python
def __init__(self, config):
    self.alpha = config.algorithm.get("alpha", 0.5)
    self.decay_gamma = config.algorithm.get("decay_gamma", 0.0)
    self.eps = 1e-8

def compute(self, phase_rewards, phase_mask, response_mask,
            boundaries_batch, index,
            alpha=None, decay_gamma=None, eps=None)
-> Tensor (batch, seq_len)
```

**Tham số:**
- `alpha`: float — tỷ lệ local vs global. `alpha=None` → dùng `self.alpha`
- `decay_gamma`: float — in-phase decay. `0.0` = không decay (hard assignment)

**Luồng** — copy từ `adpo_algorithm.py`:
1. `compute_phase_advantages(phase_rewards, phase_mask, index, eps)` → `phase_adv`
   Code: `adpo_algorithm.py:751-795`
2. `build_phase_mask(boundaries_batch, seq_len, response_mask)` → `phase_ids`
   Code: `adpo_algorithm.py:647-674`
3. `assign_phase_advantages_to_tokens(phase_adv, phase_ids, response_mask, decay_gamma, boundaries_batch)` → `token_adv`
   Code: `adpo_algorithm.py:798-888`
4. Normalize by n_phases:
   ```python
   for b in range(batch_size):
       n_phases = len(boundaries_batch[b])
       if n_phases > 0:
           token_adv[b] /= n_phases
   ```

**Lưu ý về alpha:** Hiện tại `compute_phase_advantages()` trong code cũ KHÔNG có alpha — nó dùng adaptive lambda. Kế hoạch refactor thêm alpha mới để:
```
A_final = alpha * A_local + (1 - alpha) * A_global
```
Đây là thay đổi mới so với code cũ. Cần import/copy `compute_local_advantages` và `compute_global_advantages` từ `adpo_algorithm.py:681-748` rồi kết hợp với alpha.

---

### C3. `trainer.py` — Unified Trainer

```python
class ADPOTrainer:
    def __init__(self, splitter, reward_computer, advantage_computer, config):
        self.splitter = splitter
        self.reward_computer = reward_computer
        self.advantage_computer = advantage_computer
        self.alpha = config.algorithm.get("alpha", 0.5)

    def compute_advantages(self, data) -> torch.Tensor:
        ...
```

**Luồng `compute_advantages`** — tái dùng pattern từ `adpo_trainer.py:359-991`:
1. Extract batch fields từ `data`:
   - `input_ids`, `response_mask`, `log_probs`, `index`
   - `questions`, `golden_answers`, `data_sources`, `full_responses`
   - Code tham khảo: `adpo_trainer.py:100-230`
2. Tính entropy: `compute_token_entropy(log_probs, response_mask)`
   - Code: `adpo_algorithm.py:431-464` (cũng có ở `pure_entropy_algorithm.py:128-151`)
3. Split: `self.splitter.split(entropy, response_mask, log_probs, input_ids, tokenizer)` → `boundaries`
4. Extract phase texts: `self.splitter.extract_phase_texts(...)` → `phase_texts_batch`
5. Pre-compute `outcome_rewards` bằng `compute_score`
6. `self.reward_computer.compute(boundaries, response_mask, index, ...)` → `(phase_rewards, phase_mask, metadata)`
7. `self.advantage_computer.compute(phase_rewards, phase_mask, response_mask, boundaries, index)` → `token_advantages`
8. Post-process nếu StitcherReward: `compute_stitched_advantages(token_advantages, response_mask, metadata["splice_results"], ...)`
   Code: `trajectory_stitching.py:545-603`
9. Return `token_advantages`

---

## PHASE D — Tests

### D1. Test PhaseSplitter

**File:** `tests/test_phase_splitters.py`

```python
# Fixture: tạo tensor giả
def make_batch(batch_size, seq_len, entropy_values, mask_start=0):
    entropy = torch.zeros(batch_size, seq_len)
    # điền entropy_values vào vị trí response
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, mask_start:] = 1
    return entropy, response_mask

# Test 1: Output đúng type
def test_split_returns_list_of_lists():
    splitter = PureEntropySplitter(config)
    boundaries = splitter.split(entropy, response_mask)
    assert isinstance(boundaries, list)
    assert all(isinstance(b, list) for b in boundaries)

# Test 2: Boundaries nằm trong range hợp lệ
def test_boundaries_in_range():
    boundaries = splitter.split(entropy, response_mask)
    for b_idx, bounds in enumerate(boundaries):
        active = response_mask[b_idx].nonzero()[0]
        start, end = active[0].item(), active[-1].item() + 1
        assert bounds[0] == start
        assert all(start <= bd < end for bd in bounds)

# Test 3: Số phases không vượt max_phases
def test_max_phases_respected():
    boundaries = splitter.split(entropy, response_mask)
    assert all(len(b) <= config.phase_max_K for b in boundaries)

# Test 4: Khoảng cách tối thiểu giữa boundaries
def test_min_phase_len():
    boundaries = splitter.split(entropy, response_mask)
    for bounds in boundaries:
        for i in range(1, len(bounds)):
            assert bounds[i] - bounds[i-1] >= config.phase_min_len

# Test 5: DeliEntropySplitter — think boundary
def test_deli_finds_think_boundary():
    # Tạo token_ids có </think> token
    ...
    boundaries = splitter.split(entropy, response_mask, token_ids=ids, tokenizer=tok)
    # think_end phải xuất hiện trong boundaries
    assert think_end_pos in boundaries[0]
```

### D2. Test RewardComputer

**File:** `tests/test_reward_computers.py`

```python
# Kiểm tra interface contract (quan trọng nhất)
def test_compute_returns_triple():
    result = reward.compute(boundaries_batch, response_mask, index, ...)
    assert len(result) == 3
    phase_rewards, phase_mask, metadata = result
    assert isinstance(phase_rewards, torch.Tensor)
    assert isinstance(phase_mask, torch.Tensor)
    assert isinstance(metadata, dict)

# Kiểm tra shape
def test_compute_shape():
    phase_rewards, phase_mask, _ = reward.compute(...)
    batch_size = response_mask.shape[0]
    max_K = max(len(b) for b in boundaries_batch)
    assert phase_rewards.shape == (batch_size, max_K)
    assert phase_mask.shape == (batch_size, max_K)

# Kiểm tra mask đúng
def test_phase_mask_correct():
    phase_rewards, phase_mask, _ = reward.compute(...)
    for b, bounds in enumerate(boundaries_batch):
        n_phases = len(bounds)
        assert phase_mask[b, :n_phases].all()        # active phases = 1
        assert not phase_mask[b, n_phases:].any()    # padding = 0

# EntropyReward: sum(rewards) ≈ R_total
def test_entropy_reward_sum_equals_total():
    phase_rewards, phase_mask, _ = entropy_reward.compute(
        ..., outcome_rewards=[1.0, 0.0, 1.0, 0.0]
    )
    for b in range(batch_size):
        n = int(phase_mask[b].sum().item())
        expected = 1.0 if outcome_rewards[b] >= 1.0 else -1.0
        actual = phase_rewards[b, :n].sum().item()
        assert abs(actual - expected) < 1e-4
```

### D3. Test PhaseAdvantageComputer

**File:** `tests/test_advantage_computer.py`

```python
# Shape đúng
def test_output_shape():
    token_adv = adv_computer.compute(phase_rewards, phase_mask, response_mask, boundaries, index)
    assert token_adv.shape == response_mask.shape

# Non-response tokens = 0
def test_non_response_tokens_zero():
    token_adv = adv_computer.compute(...)
    assert (token_adv * (1 - response_mask)).abs().max() < 1e-6

# alpha=0 → chỉ global
def test_alpha_zero_is_pure_global():
    adv_global = adv_computer.compute(..., alpha=0.0)
    # Tất cả responses trong cùng group nên có cùng token_adv pattern
    # (chỉ phụ thuộc vào response score, không phụ thuộc phase distribution)

# alpha=1 → chỉ local
def test_alpha_one_is_pure_local():
    adv_local = adv_computer.compute(..., alpha=1.0)
    # Responses giống nhau trong group vẫn có adv khác nhau (do phases khác nhau)

# Verify: giống output của compute_adpo_phase_advantages() từ code cũ
def test_matches_old_implementation():
    old_adv = compute_adpo_phase_advantages(
        log_probs, phase_rewards, phase_mask, response_mask, index, boundaries
    )
    new_adv = adv_computer.compute(phase_rewards, phase_mask, response_mask, boundaries, index, alpha=0.5)
    # Không cần exact match (alpha mới vs adaptive lambda cũ) nhưng shape và sign phải gần nhau
```

### D4. Test End-to-End Combo

**File:** `tests/test_trainer_combos.py`

```python
# Combo 1: (DeliEntropySplitter, JudgeReward, PhaseAdvantageComputer)
# Combo 2: (PureEntropySplitter, EntropyReward, PhaseAdvantageComputer)
# Combo 3: (PureEntropySplitter, AttentionReward, PhaseAdvantageComputer)

def test_combo_runs_without_error(splitter, reward_computer, adv_computer):
    trainer = ADPOTrainer(splitter, reward_computer, adv_computer, config)
    # Tạo fake data batch
    data = make_fake_data_batch(batch_size=4, seq_len=64, n_phases=3)
    token_adv = trainer.compute_advantages(data)
    assert token_adv.shape == (4, 64)
    assert not torch.isnan(token_adv).any()
    assert not torch.isinf(token_adv).any()
```

---

## Thứ Tự Thực Hiện

```
A (fix bugs)
    ↓
B1 + B2 (hoàn thiện EntropyReward + AttentionReward return type)
    ↓
C2 (PhaseAdvantageComputer) ← đơn giản nhất, không phụ thuộc gì mới
    ↓
D1 + D2 + D3 (tests cho splitter, reward, advantage)
    ↓
C1 (StitcherReward) + sửa TrajectoryStitcher ← phức tạp nhất
    ↓
C3 (Trainer) ← wiring tất cả lại
    ↓
D4 (end-to-end test)
```

---

## Mapping Code Cũ → Code Mới

| Code mới | Lấy từ đâu |
|----------|-----------|
| `PureEntropySplitter.split()` | `pure_entropy_algorithm.py:31-121` |
| `DeliEntropySplitter.split()` | `adpo_algorithm.py:264-576` (detect_phase_boundaries_adaptive/entropy) |
| `PhaseSplitter.extract_phase_texts()` | `adpo_algorithm.py:625-644` (segment_response_into_phases) |
| `AttentionReward.compute()` | `pure_entropy_trainer.py:437-493` |
| `JudgeReward.compute()` | `adpo_trainer.py:750-930` |
| `EntropyReward.compute()` | `entropy_credit_trainer.py:250-285` |
| `PhaseAdvantageComputer.compute()` | `adpo_algorithm.py:681-947` |
| `compute_stitched_advantages()` | `trajectory_stitching.py:545-603` (giữ nguyên) |
| Helper: `compute_token_entropy()` | `adpo_algorithm.py:431` hoặc `pure_entropy_algorithm.py:128` |
| Helper: `build_phase_mask()` | `adpo_algorithm.py:647` |
