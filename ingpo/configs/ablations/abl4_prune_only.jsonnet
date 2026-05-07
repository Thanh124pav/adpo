// Abl 4: Prune only, no ValueShare.
{ ingpo+: { enable_share: false, enable_prune: true } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_enable_share: false, ingpo_enable_prune: true },
  },
}
