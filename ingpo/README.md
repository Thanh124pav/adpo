# InGPO: Information-Gated Policy Optimization

Implementation of **InGPO** as specified in [`PLAN.md`](./PLAN.md): an
extension to **SPO (Segment Policy Optimization)** that adds two online
triggers — **ValueShare** for redundant segments and **Prune** for
information-irrelevant segments — gated by a TV bound over a per-problem
answer set `Y`.

This repo wraps the upstream [SPO](https://github.com/AIFrameResearch/SPO)
codebase: SPO is cloned verbatim into [`spo/`](./spo) and InGPO components
are layered on top via Python registration decorators and Jsonnet config
inheritance, so all SPO behaviour (phases, probability mask, replay buffer,
PPO-with-prob-mask trainer) is reused unchanged.

## Layout

```
ingpo/
├── PLAN.md                      # the spec
├── README.md
├── spo/                         # cloned upstream SPO
├── ingpo_src/
│   ├── ingpo_main.py            # entrypoint that registers ext + runs SPO
│   └── ingpo_ext/
│       ├── core/                # math: log-prob matrix, BST, TV, η/τ, triggers
│       ├── inference_strategies/ingpo_inference_strategy.py
│       └── episode_generators/ingpo_episode_generator.py
├── configs/
│   ├── ingpo_defaults.libsonnet
│   ├── polIter_*_ingpo_tree_*.jsonnet
│   ├── episode_generators/branch_factor_{444,666,888}.jsonnet
│   ├── ablations/abl{1..7}_*.jsonnet
│   └── baselines/{spo_tree,ppo,rft}_*.jsonnet
├── scripts/
│   ├── setup.sh                 # clone + pip install + dataset prep
│   ├── start_vllm_server.sh     # alias of SPO's
│   ├── train_ingpo_tree_{MATH,GSM8K}.sh
│   ├── run_baseline.sh          # SPO / PPO / RFT
│   ├── run_exp1_sample_efficiency.sh
│   ├── run_exp2_prune_share_rate.sh
│   ├── run_exp3_overhead.sh
│   ├── run_ablations.sh
│   ├── evaluate.sh
│   ├── build_global_Y.py        # Abl 5 helper
│   └── oracle_audit.py          # Abl 7 helper
└── tests/                       # 17 unit tests on the core
```

## Algorithm map

| PLAN.md section | InGPO module |
|---|---|
| Def 2.1 LogP Matrix | `ingpo_ext/core/log_prob_matrix.py` |
| Def 2.2 AvgLP & TV | `ingpo_ext/core/tv_distance.py` |
| Def 2.3 / Lemma 2.4 thresholds | `ingpo_ext/core/thresholds.py` |
| Build Y at depth=1 | `ingpo_ext/core/answer_set.py` |
| Score `log π(y_i \| traj(s))` | `ingpo_ext/core/{lp_scorer,vllm_scorer}.py` |
| BST keyed by AvgLP_K | `ingpo_ext/core/segment_index.py` |
| Online Share / Prune | `ingpo_ext/core/triggers.py` |
| Tree builder | `ingpo_ext/inference_strategies/ingpo_inference_strategy.py` |
| Edge → episode | `ingpo_ext/episode_generators/ingpo_episode_generator.py` |

## How it integrates with SPO

* **Inference strategy** `InGPOInferenceStrategy` (registered as
  `type: 'ingpo'`) subclasses SPO's `HybridInferenceStrategy` and overrides
  only `_construct_tree`. The full SPO width-W parallel expansion is kept;
  every freshly-generated child is annotated with
  `ingpo_action ∈ {expand, share, prune}` and either recursed into or
  short-circuited.
* **Episode generator** `InGPOEpisodeGenerator` (registered as
  `type: 'ingpo_episode_generator'`) subclasses
  `HybridEpisodeGenerator` and overrides only `extract_edges_from_tree`,
  which:
  - drops PRUNE edges (or zeros their advantage if
    `ingpo_zero_advantage_when_pruned`),
  - resolves SHARE edges' value/reward from the share target,
  - propagates Share/Prune metadata for wandb logging.
* **PPO trainer** is **unchanged**. The `use_prob_mask=true` setting from
  `polIter_qwen1_5b_base_spo_chain_MATH.jsonnet` is inherited so the masked
  whitening behaviour is identical.

## Quick start

```sh
# 1. Bootstrap: clone SPO, install deps, download datasets.
bash scripts/setup.sh

# 2. Start a vLLM server (in another terminal).
bash scripts/start_vllm_server.sh Qwen/Qwen2.5-1.5B 8000 42 32 0
export APP_OPENAI_VLLM_API_BASE=http://127.0.0.1:8000/v1

# 3. Train.
INGPO_TREE=666 bash scripts/train_ingpo_tree_MATH.sh

# 4. Evaluate.
bash scripts/evaluate.sh polIter_qwen1_5b_base_ingpo_tree_MATH \
    experiments/ingpo-tree-666-qwen1.5b-math/iteration_0010
```

## Reproducing the paper experiments

```sh
# Exp 1 — sample efficiency on Qwen-MATH and Rho-GSM8K, 4-4-4/6-6-6/8-8-8.
TREES="444 666 888" MODELS="qwen rho" bash scripts/run_exp1_sample_efficiency.sh

# Exp 2 — prune/share rate per depth + advantage variance.
INGPO_TREE=666 bash scripts/run_exp2_prune_share_rate.sh

# Exp 3 — overhead vs SPO.
INGPO_TREE=666 NUM_ITER=10 bash scripts/run_exp3_overhead.sh
```

## Ablations (PLAN.md §5)

```sh
ABLATIONS="abl1 abl2 abl3 abl4 abl5 abl6 abl7" bash scripts/run_ablations.sh
```

| Run name | Toggle |
|---|---|
| `abl1-K{5,20}-m{50,200}` | K vs m grid |
| `abl2-eta-{0.005..0.05}` | η theory vs grid |
| `abl3-share-{parent,root}` | Share target |
| `abl4-{share-only,prune-only,neither}` | Trigger ablation |
| `abl5-y-per-dataset` | Y precomputed once (use `scripts/build_global_Y.py`) |
| `abl6-no-dkw` | DKW band off → τ = η |
| `abl7-oracle-record` | Keep PRUNE/SHARE edges for `scripts/oracle_audit.py` |

## Tests

```sh
PYTHONPATH=ingpo_src python -m pytest tests/ -q
```

17 unit tests cover the BST, log-prob matrix, TV computation, threshold
formulas, and the trigger state machine on stub LP vectors.

## Notes

* The vLLM `/v1/completions` endpoint is hit with `echo=True, logprobs=1,
  max_tokens=0` to score `log π(y_i | traj(s))` — see
  `ingpo_ext/core/vllm_scorer.py`.
* `K=10, m=100, ε=0.02` are the defaults from PLAN.md §3-4 and live in
  `configs/ingpo_defaults.libsonnet`.
* All compute and tree-building knobs (W, D, M, temperatures, prompt
  templates, reward function, replay buffer policy) come from the inherited
  SPO config — no SPO file is modified.

## License

Same as SPO (MIT). See [`spo/LICENSE`](./spo/LICENSE).
