// Reusable overlay that converts an SPO-tree config into an InGPO-tree
// config. Apply via:
//
//   (import '../spo/configs/polIter_<model>_<dataset>_spo_tree.jsonnet')
//   + (import 'ingpo_defaults.libsonnet')
//   + (import 'ingpo_overlay.libsonnet')
//
// The overlay reads `$.ingpo.*` (provided by ingpo_defaults.libsonnet)
// and mirrors the knobs onto the inference strategy / episode generator
// where SPO's main entrypoint will pass them to InGPO's __init__.
//
// Anything specific to a single experiment (eta, K, m, prune-only, ...)
// is supplied by ablation snippets that override `$.ingpo.*` BEFORE this
// overlay is applied. Because Jsonnet evaluates everything lazily, the
// final values come out the way the caller chained them.
{
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
    ingpo_demo_examples_per_tree: $.ingpo.demo_examples_per_tree,
    ingpo_demos_dir: $.ingpo.demos_dir,
    ingpo_log_demos_to_wandb: $.ingpo.log_demos_to_wandb,
  },
}
