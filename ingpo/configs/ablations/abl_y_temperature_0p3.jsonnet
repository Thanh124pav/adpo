// Y-generation temperature sweep: low (0.3) — Y is more peaked.
{ ingpo+: { y_temperature: 0.3 } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_y_temperature: 0.3 },
  },
}
