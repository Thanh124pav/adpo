// TPO deep-tree variant: larger K, deeper trees, adaptive token budget.
//
// Use this when the model is stronger and benefits from deeper exploration
// before the P-degradation criterion cuts off a branch.
//
// Usage:
//   APP_SEED=42 \
//   TPO_SERVER_URL=http://localhost:8000/v1 \
//   TPO_MODEL_NAME=Qwen/Qwen2.5-1.5B \
//   python -m treetune.main \
//       ../tpo/configs/polIter_qwen1_5b_tpo_MATH_deep_tree.jsonnet

(import 'polIter_qwen1_5b_tpo_MATH.jsonnet')
+ {
  episode_generator+: {
    inference_strategy+: {
      // Deeper trees
      termination_strategy+: {
        max_degradations: 5,   // K = 5 (more lenient)
        max_depth: 12,
      },

      // Adaptive token budget: 1024→128 tokens as depth grows
      segmentation_strategy: {
        type: 'adaptive_token',
        initial_max_tokens: 1024,
        min_max_tokens: 128,
        decay_per_depth: 0.7,
        temperature: 0.6,
        top_p: 1.0,
      },
    },
  },
}
