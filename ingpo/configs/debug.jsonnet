// Debug overlay: shrink iterations / tree size for quick smoke tests.
// Stack as the *last* config:
//   --configs ...,${INGPO_ROOT}/configs/debug.jsonnet
{
  num_iterations: 2,
  num_episodes_per_iteration: 16,

  episode_generator+: {
    dataset_num_samples_per_iteration: 4,
    inference_strategy+: {
      max_depth: 2,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 2 },
          { depth: 1, branch_factor: 2 },
        ],
      },
      // Keep Y small so we don't waste time generating answers we'll throw away.
      ingpo_m: 8,
      ingpo_K: 4,
    },
  },

  ingpo+: { K: 4, m: 8 },
}
