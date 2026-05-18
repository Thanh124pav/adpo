// VinePPO — PPO with vine-style value estimation.
//
// Same PPO trainer; the episode generator samples short branched rollouts
// at every reasoning step and uses them as a non-parametric value baseline,
// removing the need for a learned value head.
{
  episode_generator+: {
    type: 'vineppo_episode_generator',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: {
    type: 'ppo',
    params+: { use_prob_mask: false },
  },
}
