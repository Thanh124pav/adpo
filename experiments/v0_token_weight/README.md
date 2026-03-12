# V0: Token-Level Weighting (Deprecated)

Original approach: weight each token by its log-probability uncertainty.

    w_t = (-log pi(a_t | s_t))^beta / sum(-log pi)^beta
    A_i(t) = w_t * A_i

This was replaced by the phase-based decomposition approach (v1)
which segments responses into phases and uses LLM-as-Judge for
per-phase reward scoring.

See the main `adpo/` directory for the current implementation.
