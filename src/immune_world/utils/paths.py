"""Dataset / checkpoint path resolver.

Resolution order:
    1. Explicit arg to `resolve_dataset_path`
    2. Environment variable `IMMUNE_WORLD_DATA_DIR`
    3. `./data/processed/<name>`
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_dataset_path(name: str, override: str | os.PathLike[str] | None = None) -> Path:
    """Return the canonical on-disk path for dataset `name`.

    Does NOT create the directory; caller must check `.exists()` before reading.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.environ.get("IMMUNE_WORLD_DATA_DIR")
    root = Path(env).expanduser() if env else Path("data/processed")
    return (root / name).resolve()
