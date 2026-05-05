// TPO with fixed branching (ablation config).
//
// Identical to the main TPO config but replaces EntropyBranchingStrategy
// with FixedBranchingStrategy (branch_factor=3), isolating the effect of
// dynamic branching in experiments.
//
// Usage:
//   APP_SEED=42 \
//   TPO_SERVER_URL=http://localhost:8000/v1 \
//   TPO_MODEL_NAME=Qwen/Qwen2.5-1.5B \
//   python -m treetune.main \
//       ../tpo/configs/polIter_qwen1_5b_tpo_MATH_fixed_branch.jsonnet

(import 'polIter_qwen1_5b_tpo_MATH.jsonnet')
+ {
  episode_generator+: {
    inference_strategy+: {
      // Replace entropy branching with fixed 3-way branching (SPO-compatible)
      branching_strategy: {
        type: 'fixed',
        branch_factor: 3,
      },
    },
  },
}
