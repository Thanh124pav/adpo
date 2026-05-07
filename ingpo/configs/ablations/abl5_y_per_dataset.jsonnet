// Abl 5 V1: Y precomputed once per dataset (rather than per problem).
// We approximate this by setting the answer-set field to read from a static
// JSONL pre-built offline; see scripts/build_global_Y.py.
{ ingpo+: { y_field: 'global_Y' } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_y_field: 'global_Y' },
  },
}
