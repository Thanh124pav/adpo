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
│   ├── ingpo_overlay.libsonnet            # SPO-tree -> InGPO-tree overlay
│   ├── debug.jsonnet                      # tiny smoke-test config
│   ├── polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet
│   ├── polIter_qwen1b_ingpo_tree_{MATH,point24}.jsonnet
│   ├── polIter_qwen05b_ingpo_tree_GSM8K.jsonnet
│   ├── polIter_rho1bSft2_ingpo_tree_{MATH,GSM8K}.jsonnet
│   ├── episode_generators/
│   │   ├── branch_factor_{333,444,456,555,654,666,777,888}.jsonnet
│   │   ├── branch_factor_{3333,4444}.jsonnet     # D=4
│   │   ├── branch_factor_33333.jsonnet           # D=5
│   │   └── depth_2_W6.jsonnet                    # D=2 control
│   ├── ablations/
│   │   ├── abl1_K{1,5,10,20}_m{20,50,100,200,500}.jsonnet
│   │   ├── abl2_eta_{0p005,0p01,0p02,0p05,0p10,0p20}.jsonnet
│   │   ├── abl3_share_target_{parent,root}.jsonnet
│   │   ├── abl4_{share-only,prune-only,neither}.jsonnet
│   │   ├── abl{5,6,7}_*.jsonnet
│   │   └── abl_y_temperature_{0p3,1p0}.jsonnet
│   └── baselines/
│       ├── spo_{tree,chain}_{MATH,GSM8K,...}.jsonnet
│       ├── {ppo,grpo,dpo_positive,restem}_{MATH,GSM8K}.jsonnet
│       ├── vineppo_GSM8K.jsonnet
│       └── rft_MATH.jsonnet
├── scripts/
│   ├── setup.sh                          # clone + pip install + dataset prep
│   ├── start_vllm_server.sh              # alias of SPO's
│   ├── download_cached.sh                # alias of SPO's cache helper
│   ├── train_ingpo_tree_{MATH,GSM8K}.sh
│   ├── train_ingpo_tree_{qwen1b_MATH,qwen1b_point24,qwen05b_GSM8K,rho_MATH}.sh
│   ├── train_debug.sh                    # 2-iter end-to-end smoke run
│   ├── run_baseline.sh                   # SPO / PPO / GRPO / DPO / ReSTEM / VinePPO / RFT
│   ├── run_seeds.sh                      # multi-seed driver
│   ├── run_smoke.sh                      # config compile + unit tests (CPU-only)
│   ├── run_all_models.sh                 # iterate every (model, dataset) pair
│   ├── run_exp1_sample_efficiency.sh
│   ├── run_exp2_prune_share_rate.sh
│   ├── run_exp3_overhead.sh
│   ├── run_exp_deep_tree.sh              # D in {2,3,4,5} compute Pareto
│   ├── run_exp_eta_sweep.sh              # eta soundness/throughput trade-off
│   ├── run_exp_K_m_sweep.sh              # scoring budget trade-off
│   ├── run_ablations.sh
│   ├── evaluate.sh                       # SPO/InGPO checkpoint eval
│   ├── evaluate_long_cot.sh              # lighteval recipe (math_500 / gsm8k)
│   ├── inspect_tree.py                   # pretty-print one InGPO tree JSON
│   ├── aggregate_stats.py                # episodes/*.json -> stats.csv
│   ├── build_global_Y.py                 # Abl 5 helper
│   └── oracle_audit.py                   # Abl 7 helper
└── tests/                                # 17 unit tests on the core
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

For a CPU-only smoke check (compiles every jsonnet config + runs the unit
tests, no GPU/vLLM needed):

```sh
bash scripts/run_smoke.sh
```

For an end-to-end debug run (2 iterations, depth-2 tree, m=8) once vLLM is
up:

```sh
bash scripts/train_debug.sh
```

## Logging (offline-friendly)

InGPO logging works **without internet / wandb** — demos and rates are
written to local files under
`<APP_DIRECTORY>/<exp_name>/ingpo_demos/` per tree. wandb is optional
(`ingpo.log_demos_to_wandb = false` by default, so no upload attempts).

### Per-tree scalar metrics (always emitted, also to wandb if enabled)

Aggregate rates:

* `ingpo/share_rate`, `ingpo/prune_rate`,
  `ingpo/expanded_count`, `ingpo/shared_count`, `ingpo/pruned_count`
* `ingpo/avg_tv_when_share`, `ingpo/avg_gap_when_prune`
* `ingpo/answer_set_size`

Per-depth breakdown (one set per depth `d` reached in the tree):

* `ingpo/depth_<d>/n`, `expand_count`, `share_count`, `prune_count`,
  `share_rate`, `prune_rate`

### Demo files (local filesystem)

Two append-only files written next to your experiment dir:

| File                                     | Audience            | Format                         |
|------------------------------------------|---------------------|--------------------------------|
| `ingpo_demos/demos.jsonl`                | scripts / analysis  | JSON-Lines, one record/tree    |
| `ingpo_demos/demos.md`                   | humans              | Markdown, section per tree     |

Each tree contributes up to `ingpo.demo_examples_per_tree` SHARE rows and
the same for PRUNE rows. Each row carries:

| column        | meaning                                                  |
|---------------|----------------------------------------------------------|
| question_id   | dataset row id                                           |
| action        | `share` or `prune`                                       |
| depth         | depth of the child segment                               |
| seg_id        | tree-internal id of the child segment                    |
| parent_text   | text of the parent segment (truncated to 240 chars)      |
| child_text    | text of the candidate segment that fired the trigger     |
| target_text   | SHARE only — text of the segment we shared into          |
| target_seg_id | id of that share target                                  |
| avg_lp_K      | the candidate's K-subset AvgLP                           |
| tv_m          | TV_m to share target (SHARE only)                        |
| gap_m         | AvgLP_m gap to parent (PRUNE only)                       |
| eta, tau      | thresholds in effect for this decision                   |

### Reading demos on a server with no GUI

Watch the human-readable Markdown live during training:

```sh
bash scripts/tail_demos.sh <exp_name>           # tail -F demos.md
bash scripts/tail_demos.sh <exp_name> jsonl     # raw JSONL stream
```

Filter / pretty-print after the run:

```sh
# show first 20 share+prune demos
python scripts/inspect_demos.py experiments/<exp>/ingpo_demos/demos.jsonl

# only PRUNE demos at depth 2, first 5
python scripts/inspect_demos.py experiments/<exp>/ingpo_demos/demos.jsonl \
    --action prune --depth 2 --limit 5

# just totals, no demo bodies
python scripts/inspect_demos.py experiments/<exp>/ingpo_demos/demos.jsonl --summary
```

`inspect_demos.py` only needs the standard library — copy it onto any
offline machine.

### Knobs (override in any jsonnet via `ingpo+: { ... }`)

| Knob                             | Default | Purpose                                |
|----------------------------------|---------|----------------------------------------|
| `demo_examples_per_tree`         | `2`     | rows of each (SHARE/PRUNE) per tree    |
| `demos_dir`                      | `null`  | absolute override; else `<exp_root>/ingpo_demos` |
| `log_demos_to_wandb`             | `false` | also push the wandb.Table when running with wandb |

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
