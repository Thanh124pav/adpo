// Abl 1 row: K=1, m=20 (cheapest scoring budget). Gates on a single
// answer index — useful to stress-test variance of the AvgLP_K signal.
{ ingpo+: { K: 1, m: 20 } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_K: 1, ingpo_m: 20 },
  },
}
