"""Wrapper entrypoint that registers InGPO components, then defers to SPO.

Used by every script under `scripts/` so the InGPO inference strategy and
episode generator are visible in the global registry before treetune.main
parses configs.
"""

from __future__ import annotations

import sys


def main() -> None:
    import ingpo_ext

    ingpo_ext.register_with_treetune()

    # Hand off to SPO main.
    from treetune.main import EntryPoint  # type: ignore
    import fire  # type: ignore

    fire.Fire(EntryPoint)


if __name__ == "__main__":
    main()
