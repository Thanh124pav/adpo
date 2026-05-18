// Very deep tree: D=5, W=4 at every level.
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 5,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 4 },
          { depth: 1, branch_factor: 4 },
          { depth: 2, branch_factor: 4 },
          { depth: 3, branch_factor: 4 },
          { depth: 4, branch_factor: 4 },
        ],
      },
    },
  },
}
