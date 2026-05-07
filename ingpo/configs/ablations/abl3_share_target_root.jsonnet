// Abl 3 V2: Share with root only.
{ ingpo+: { share_target: 'root' } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_share_target: 'root' },
  },
}
