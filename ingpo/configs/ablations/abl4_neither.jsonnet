// Abl 4 sanity row: triggers off ⇒ should match SPO-tree exactly.
{ ingpo+: { enable_share: false, enable_prune: false } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_enable_share: false, ingpo_enable_prune: false },
  },
}
