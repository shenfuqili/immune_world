"""Module entry point: `python -m immune_world` → training CLI."""

from __future__ import annotations

from immune_world.cli.train import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
