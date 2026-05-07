// Abl 4: ValueShare only, no Prune.
{ ingpo+: { enable_share: true, enable_prune: false } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_enable_share: true, ingpo_enable_prune: false },
  },
}
