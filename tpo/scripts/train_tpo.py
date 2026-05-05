"""
TPO Training Entry Point
========================

Registers TPO's inference strategy and episode generator with treetune's
registry, then delegates to SPO's standard ``treetune.main`` entry point.

Usage (from the repo root)
--------------------------

Single GPU (debug / quick test):

    TPO_SERVER_URL=http://localhost:8000/v1 \\
    TPO_MODEL_NAME=Qwen/Qwen2.5-1.5B \\
    APP_SEED=42 \\
    python tpo/scripts/train_tpo.py \\
        tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet

Multi-GPU with Accelerate (mirrors SPO's launch pattern):

    TPO_SERVER_URL=http://localhost:8000/v1 \\
    TPO_MODEL_NAME=Qwen/Qwen2.5-1.5B \\
    APP_SEED=42 \\
    accelerate launch --num_processes 4 tpo/scripts/train_tpo.py \\
        tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet

Config variants:
    polIter_qwen1_5b_tpo_MATH.jsonnet             # default (entropy branch + P-degradation)
    polIter_qwen1_5b_tpo_MATH_deep_tree.jsonnet   # larger K, deeper trees
    polIter_qwen1_5b_tpo_MATH_fixed_branch.jsonnet # ablation: fixed branching

Environment variables
---------------------
TPO_SERVER_URL   Required. vLLM server URL, e.g. ``http://localhost:8000/v1``.
TPO_MODEL_NAME   Required. Model identifier registered on the vLLM server.
APP_SEED         Optional. Integer random seed (default: 42).
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Ensure the project roots are on sys.path so treetune and tpo are both
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
# 3. Validate required environment variables
# ---------------------------------------------------------------------------
for _env_var in ("TPO_SERVER_URL", "TPO_MODEL_NAME"):
    if not os.environ.get(_env_var):
        print(f"[train_tpo] ERROR: Environment variable {_env_var} is not set.", flush=True)
        print(f"  Example: {_env_var}=http://localhost:8000/v1", flush=True)
        sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Hand off to SPO's standard main entry point (fire.Fire → EntryPoint).
# ---------------------------------------------------------------------------
from treetune.main import EntryPoint
import fire

if __name__ == "__main__":
    fire.Fire(EntryPoint)
