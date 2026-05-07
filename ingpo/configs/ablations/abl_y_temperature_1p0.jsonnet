// Y-generation temperature sweep: high (1.0) — Y is more diverse.
{ ingpo+: { y_temperature: 1.0 } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_y_temperature: 1.0 },
  },
}
