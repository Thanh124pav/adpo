// InGPO-tree on MATH with Qwen2.5-1.5B-base.
// Inherits the SPO-tree config and overrides:
//   * episode_generator.type   -> ingpo_episode_generator
//   * inference_strategy.type  -> ingpo
//   * adds InGPO knobs (K, m, eta, etc.)
//
// Run from the InGPO project root with both SPO and InGPO configs on the
// jsonnet include path:
//   jsonnet -J ../spo -J . polIter_qwen1_5b_base_ingpo_tree_MATH.jsonnet

(import '../spo/configs/polIter_qwen1_5b_base_spo_tree_MATH.jsonnet')
+ (import 'ingpo_defaults.libsonnet')
+ {
  episode_generator+: {
    type: 'ingpo_episode_generator',
    inference_strategy+: {
      type: 'ingpo',

      // -- InGPO knobs (mirrored on inference strategy) --
      ingpo_K: $.ingpo.K,
      ingpo_m: $.ingpo.m,
      ingpo_epsilon: $.ingpo.epsilon,
      ingpo_r_max: $.ingpo.r_max,
      ingpo_alpha: $.ingpo.alpha,
      ingpo_use_dkw: $.ingpo.use_dkw,
      ingpo_eta_override: $.ingpo.eta_override,
      ingpo_enable_share: $.ingpo.enable_share,
      ingpo_enable_prune: $.ingpo.enable_prune,
      ingpo_share_target: $.ingpo.share_target,
      ingpo_y_temperature: $.ingpo.y_temperature,
      ingpo_y_max_tokens: $.ingpo.y_max_tokens,
      ingpo_y_field: $.ingpo.y_field,
      ingpo_score_concurrency: $.ingpo.score_concurrency,
    },

    // -- InGPO knobs on episode generator --
    ingpo_zero_advantage_when_pruned: $.ingpo.zero_advantage_when_pruned,
    ingpo_emit_pruned_edges: $.ingpo.emit_pruned_edges,
  },
}
