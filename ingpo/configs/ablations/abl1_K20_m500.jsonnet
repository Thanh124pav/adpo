// Abl 1 row: K=20, m=500 (large scoring budget — quasi-oracle TV).
{ ingpo+: { K: 20, m: 500 } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_K: 20, ingpo_m: 500 },
  },
}
