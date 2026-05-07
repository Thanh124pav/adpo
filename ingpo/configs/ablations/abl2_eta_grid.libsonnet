// Abl 2: pin eta to a fixed value, bypassing Lemma 2.4.
function(eta) {
  ingpo+: { eta_override: eta, use_dkw: true },
}
+ {
  episode_generator+: {
    inference_strategy+: { ingpo_eta_override: eta },
  },
}
