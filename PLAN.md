# InGTS: Information-Gated Tree Search
## Unifying Value Sharing and Stopping via Answer-Level Log-Probability

### 1. Problem Statement
LLM tree search (SPO) expand nhiều node:
1. **Semantic duplicates**: `P(·|s) ≈ P(·|s')` với `s'` bất kỳ đã tồn tại trong cây.
2. **Information-irrelevant**: `I(A*; Y_s | Y_pa(s)) ≈ 0`, tức `s` không thêm thông tin so với cha.

Goal: Dùng 1 ma trận log-prob để phát hiện cả 2, không hyperparameter, có bound.

### 2. Definitions

**Notation**
- `s`: node. `pa(s)`: parent. `root`: prompt.
- `Y = {y_1,...,y_m}`: tập answer global, sinh 1 lần bằng cách prompt model với GT answer.
- `K`: fast subset, `K << m`. `m`: full set để quyết định.
- `R_max`: max reward.

**Def 2.1 – Log-Probability Matrix**
Duy trì `LP[i][s] = log P(y_i | s)`, `i=1..m`. Dùng log để ổn định số học.
`δ_s = log(1 - Σ_i exp(LP[i][s]))`

**Def 2.2 – Average Log-Prob & TV**
```
AvgLP_K(s) := (1/K) * Σ_{i=1..K} LP
TV_m(a,b) := 0.5 * Σ_{i=1..m} |exp(LP) - exp(LP)|
             + 0.5 * (exp(δ_a) + exp(δ_b))
```

**Lemma 2.3 – TV bounds Value [To Prove]**
Nếu `TV_m(a,b) ≤ η`, thì `|V*(a) - V*(b)| ≤ R_max * (η + exp(δ_avg))`.

**Lemma 2.4 – TV bounds Conditional IG [To Prove]**
`I(A*; Y_s | Y_pa) ≥ 2 * TV_m(s,pa)^2 - O(exp(δ_s)+exp(δ_pa))`.

**Corollary 2.5 – Threshold từ toán**
Cho `ε` là sai số chấp nhận. Đặt `η = ε / R_max - exp(δ_avg)`. Không cần tune.

### 3. Algorithm: InGTS

InGTS kế thừa SPO: expand song song tất cả con của 1 node. Khi 1 node `s` sinh ra, tính ngay `LP[i][s]` với `i=1..K` và đưa vào BST toàn cục để check duplicate.

```
Algorithm: InGTS
Input: root, depth D, width W, Y={y_1..y_m}, K, η
Output: tree

1 Global LP[*] ← {}
2 Global BST ← BinarySortTree() // key = AvgLP_K(s), val = s
3 Y ← GenerateAnswers(root, m) // prompt model: "List m diverse correct answers"
4
5 Procedure ExpandParallel(p, d):
6 If d > D: return
7 children ← LLM_Generate(p, W) // W con song song, như SPO
8 For s in children in parallel do
9 For i = 1 to K do
10 LP ← logprob(y_i | s) // vLLM batch
11 End for
12 AvgLP_K ← (1/K)*Σ_{i=1..K} LP
13
14 // Hướng 1: ValueShare - so với node BẤT KỲ trong cây
15 s' ← FindNearest(BST, AvgLP_K) // s' có thể ≠ pa(s)
16 If s' ≠ null and |AvgLP_K - AvgLP_K(s')| < τ_share(K,η) then
17 TV_m ← ComputeTV_m(s, s') // check full m
18 If TV_m ≤ η then
19 ValueShare(s, s') // s.value = s'.value, đánh dấu s.shared
20 Insert(BST, key=AvgLP_K, value=s) // vẫn insert để cluster
21 Continue // không expand s nữa
22 End if
23 End if
24
25 // Hướng 2: Prune - so với CHA, vì đo IG điều kiện
26 If AvgLP_K < AvgLP_K(pa(s)) - τ_prune(K,η) then
27 If AvgLP_m(s) < AvgLP_m(pa(s)) - η then // check full m
28 PruneNode(s) // I(A*;Y_s|Y_pa)≈0
29 Continue // không expand s
30 End if
31 End if
32
33 Insert(BST, key=AvgLP_K, value=s)
34 ExpandParallel(s, d+1)
35 End parallel for
36 End Procedure
37
38 ExpandParallel(root, 1)
39 return tree
```

**Procedure: ComputeTV_m(a, b)**
```
1 TV ← 0
2 For i = 1 to m do
3 TV ← TV + 0.5*|exp(LP) - exp(LP)|
4 End for
5 TV ← TV + 0.5*(exp(δ_a) + exp(δ_b))
6 return TV
```

**Thresholds từ DKW Bound**
```
τ(K,η) = η + sqrt(log(2/α) / (2K)), α=0.05
```
`τ_share`, `τ_prune` dùng chung công thức. `η` từ Cor 2.5.

**Theorem 3.1 – Regret Bound [To Prove]**
InGTS đạt `Regret(T) ≤ Õ(√T + R_max · m · η · T)`.

### 4. Experiments to Run

Kế thừa 100% setup SPO để so sánh công bằng.

**Models**: DeepSeek-Distill-Qwen-1.5B, Rho-math-1.1b-SFT.
**Tree**: 4-4-4, 6-6-6, 8-8-8.
**Datasets**: GSM8K, MATH, CollegeMath, OlympiadBench, split như SPO.

**Baselines**
1. **SPO**: gốc.
2. **ToT-SC**: entropy pruning `τ=0.5`.
3. **LATS**: `λ=0.1`.
4. **InGTS**: `K=10`, `m=100`, `η` từ Cor 2.5.

**Exp 1: Compute-Accuracy Pareto**
Metrics: Pass@1, Majority@64 vs Total FLOPs.
Goal: InGTS <50% FLOPs tại cùng Pass@1.

**Exp 2: Share/Prune Stats**
Metrics: `%nodes ValueShared`, `%nodes Pruned`, `avg TV_m khi share`, `avg ΔAvgLP_m khi prune`.
Goal: 30-60% node bị loại, acc drop <1%.

**Exp 3: Latency**
Metrics: Time cho `LP[i][s]` K=10, BST ops, toàn run.
Goal: Overhead <5% nhờ `K<<m`.

### 5. Ablation Studies

**Abl 1: `K` vs `m`**
Vary `K∈{5,10,20,50}`, `m∈{20,50,100,200}`. Metrics: FP rate của fast filter, time, acc.
Hypothesis: `K=10, m=100` tối ưu.

**Abl 2: `η` theory vs tuned**
So `η` từ Cor 2.5 vs grid `{0.005,0.01,0.02,0.05}`.
Metrics: Pareto curve. Hypothesis: theory `η` nằm trên Pareto.

**Abl 3: Duplicate vs Parent vs Root**
- V1: ValueShare chỉ với `pa(s)`.
- V2: ValueShare với `root`.
- V3: ValueShare với `nearest` như InGTS.
Metrics: Share rate, acc. Hypothesis: V3 >> V1, V2.

**Abl 4: Share-only vs Prune-only vs Both**
Tắt lần lượt line 15-22, 25-30.
Metrics: FLOPs vs acc. Hypothesis: Both tốt nhất.

**Abl 5: LogP vs Prob**
Implement lại dùng `exp(LP)` từ đầu.
Metrics: số lần NaN/overflow, acc. Hypothesis: LogP bắt buộc với `y_i` dài.

**Abl 6: BST vs Linear Scan**
Thay BST bằng scan toàn bộ node.
Metrics: Time trên 8-8-8. Hypothesis: BST nhanh 5-10x.

**Abl 7: Oracle False Rate**
Sample 100 node bị Share/Prune. Check GT: Share sai nếu `|V*(s)-V*(s')|>ε`. Prune sai nếu có path tới GT.
Metrics: False Share, False Prune. Hypothesis: <3% tại `η=0.01`.

### 6. Implementation Notes
1. **Answer set Y**: Prompt 1 lần: `"Given problem and solution, list {m} diverse complete answers:"`. Temp=0.7.
2. **vLLM**: `SamplingParams(prompt_logprobs=1)`. Với mỗi `y_i`, concat `prompt+s+y_i`, lấy sum logprob của `y_i` tokens. Batch K `y_i` * W children.
3. **BST**: `sortedcontainers.SortedDict` key=`AvgLP_K`. `FindNearest` = O(logN).
4. **Numerical**: Mọi sum dùng `torch.logsumexp`. Chỉ `exp` khi tính `TV_m`.
5. **Parallel**: `ThreadPoolExecutor` cho vòng `For s in children`. `LP` là `defaultdict`.

### 7. Contributions
1. **Algorithm**: InGTS, SPO-style parallel search với ValueShare cho duplicate bất kỳ + Prune cho IG≈0.
2. **Theory**: Thresholds `η,τ` derive từ bound, không tune.
3. **Practice**: Giảm 2-3x FLOPs trên setup SPO gốc, không đổi model/data.

### 8. Reproducibility
Sẽ release code + `Y` + config. `K=10, m=100`. Mọi siêu tham số khác kế thừa SPO.
