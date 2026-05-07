// Abl 7: keep PRUNE/SHARE edges in the dataset so the analyzer can later
// audit them against the unmodified SPO baseline.
{ ingpo+: { emit_pruned_edges: true } }
+ {
  episode_generator+: {
    ingpo_emit_pruned_edges: true,
  },
}
