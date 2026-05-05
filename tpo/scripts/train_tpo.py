"""
TPO Training Entry Point
========================

Registers TPO's inference strategy and episode generator with treetune's
registry, then delegates to SPO's standard ``treetune.main`` entry point.

vLLM lifecycle
--------------
The vLLM server is managed INTERNALLY by TPOEpisodeGenerator — it is started
at the beginning of each iteration with the CURRENT actor model checkpoint and
stopped afterwards.  This is identical to how SPO's on-policy generators work.
No external vLLM server needs to be started before running this script.

Usage (from the repo root)
--------------------------

Single GPU (debug / quick test):

    APP_SEED=42 python tpo/scripts/train_tpo.py \\
        tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet run_iteration

Multi-GPU with Accelerate (mirrors SPO's launch pattern):

    APP_SEED=42 accelerate launch \\
        --num_processes 4 \\
        --config_file tpo/configs/accelerate/default_config.yaml \\
        tpo/scripts/train_tpo.py \\
        tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet run_iteration

Config variants:
    polIter_qwen1_5b_tpo_MATH.jsonnet              # default (entropy + P-degradation)
    polIter_qwen1_5b_tpo_MATH_deep_tree.jsonnet    # K=5, depth=12, adaptive tokens
    polIter_qwen1_5b_tpo_MATH_fixed_branch.jsonnet # ablation: fixed branch_factor=3

Environment variables
---------------------
APP_SEED   Optional. Integer random seed (default: 42).  Same as SPO.
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Ensure project roots are on sys.path so both treetune and tpo are
#    importable regardless of from which directory this script is launched.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # /home/user/adpo
_SPO_SRC   = _REPO_ROOT / "spo" / "src"

for _p in [str(_REPO_ROOT), str(_SPO_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# 2. Register TPO classes with treetune BEFORE treetune.main initialises.
#    Importing tpo.training triggers:
#      - TPOInferenceStrategy → registered as "tpo" in InferenceStrategy
#      - TPOEpisodeGenerator  → registered as "tpo" in EpisodeGenerator
# ---------------------------------------------------------------------------
try:
    import tpo.training  # noqa: F401  (side-effect import for registration)
    print("[train_tpo] TPO classes registered with treetune registry.", flush=True)
except Exception as exc:
    print(f"[train_tpo] WARNING: TPO class registration failed: {exc}", flush=True)
    print("[train_tpo] Continuing — classes may already be registered.", flush=True)

# ---------------------------------------------------------------------------
# 3. Default APP_SEED if not set (SPO does the same)
# ---------------------------------------------------------------------------
if not os.environ.get("APP_SEED"):
    os.environ["APP_SEED"] = "42"
    print("[train_tpo] APP_SEED not set, defaulting to 42.", flush=True)

# ---------------------------------------------------------------------------
# 4. Hand off to SPO's standard main entry point (fire.Fire → EntryPoint).
# ---------------------------------------------------------------------------
from treetune.main import EntryPoint
import fire

if __name__ == "__main__":
    fire.Fire(EntryPoint)
