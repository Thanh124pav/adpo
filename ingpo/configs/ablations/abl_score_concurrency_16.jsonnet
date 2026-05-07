// Lower vLLM concurrency (16) — useful when running on a single small GPU.
{ ingpo+: { score_concurrency: 16 } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_score_concurrency: 16 },
  },
}
