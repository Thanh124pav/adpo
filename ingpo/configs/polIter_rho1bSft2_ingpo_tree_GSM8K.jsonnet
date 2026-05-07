// InGPO-tree on GSM8K with Rho-1.1B-SFT.
(import '../spo/configs/polIter_rho1bSft2_spo_tree_GSM8K.jsonnet')
+ (import 'ingpo_defaults.libsonnet')
+ {
  episode_generator+: {
    type: 'ingpo_episode_generator',
    inference_strategy+: {
      type: 'ingpo',
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
    ingpo_zero_advantage_when_pruned: $.ingpo.zero_advantage_when_pruned,
    ingpo_emit_pruned_edges: $.ingpo.emit_pruned_edges,
  },
}
