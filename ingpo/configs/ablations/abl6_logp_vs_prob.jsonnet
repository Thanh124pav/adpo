// Abl 6: numerical guard - the engine always works in log-space, so this
// ablation is a control: turn off DKW band so tau == eta and instrument
// NaN/overflow counts. The actual exp(LP) variant is implemented as a
// separate experimental flag handled in code (set INGPO_USE_PROB_EXP=1 in
// env to compare against the log-space default).
{ ingpo+: { use_dkw: false } }
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_use_dkw: false },
  },
}
