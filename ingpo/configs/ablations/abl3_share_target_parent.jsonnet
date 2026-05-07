// Abl 3 V1: Share with parent only.
{ ingpo+: { share_target: 'parent' } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_share_target: 'parent' },
  },
}
