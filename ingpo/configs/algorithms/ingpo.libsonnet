// InGPO — Information-Gated Policy Optimization.
//
// PPO trainer on top of SPO-tree, augmented with two online triggers:
//   * ValueShare: collapse segments whose log-prob distribution matches an
//                 already-evaluated sibling/parent.
//   * Prune:      drop segments whose value cannot exceed the parent's by
//                 more than epsilon.
// Both triggers are gated by Total Variation bounds.
//
// Compose with model/task base + ingpo defaults + overlay:
//   (import 'algorithms/ingpo.libsonnet')
//   + (import 'ingpo_defaults.libsonnet')
//   + (import 'ingpo_overlay.libsonnet')
{
  episode_generator+: {
    type: 'ingpo_episode_generator',
    inference_strategy+: { type: 'ingpo' },
  },
  trainer+: { type: 'ppo' },
}
