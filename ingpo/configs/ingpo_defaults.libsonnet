// Default InGPO hyperparameters from PLAN.md §3 / §4.
{
  ingpo: {
    // PLAN line 112: K=10, m=100
    K: 10,
    m: 100,

    // Lemma 2.4: eta = epsilon / R_max - exp(delta_avg). Setting eta_override
    // to null instructs the engine to derive it on the fly.
    epsilon: 0.02,
    r_max: 1.0,
    alpha: 0.05,
    use_dkw: true,
    eta_override: null,

    // Triggers
    enable_share: true,
    enable_prune: true,
    share_target: 'nearest',  // 'nearest' | 'parent' | 'root'

    // Y generation
    y_temperature: 0.7,
    y_max_tokens: 512,
    y_field: 'answer',

    // Tail-fill concurrency to vLLM /completions
    score_concurrency: 64,

    // Edge handling
    zero_advantage_when_pruned: true,
    emit_pruned_edges: false,
  },
}
