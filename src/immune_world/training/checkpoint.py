"""Atomic checkpoint save / load with seed persistence.

R4 — "Atomic checkpoint writes (tmp file + os.replace)" + "single `set_seed(seed)` utility; seed
stored in every checkpoint + every resumed run restores it".
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    """Save atomically: write to `<path>.tmp`, fsync, then `os.replace(tmp, path)`.

    `state` may contain a ``seed`` entry; if so it is persisted alongside the captured RNG state
    of Python / NumPy / torch (CPU + CUDA where available) so a resume reproduces exactly.
    """
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    enriched = dict(state)
    enriched.setdefault("rng", _snapshot_rng_state())

    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(enriched, tmp)
    # fsync the tmp file before the atomic rename.
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and (optionally) restore every RNG state it carries."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    state = cast(
        "dict[str, Any]", torch.load(source, map_location=map_location, weights_only=False)
    )
    if restore_rng and "rng" in state:
        _restore_rng_state(state["rng"])
    return state


def _snapshot_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(snapshot: dict[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["cuda"])
