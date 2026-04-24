"""Unit tests for atomic checkpointing + RNG-state persistence."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from immune_world.training.checkpoint import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


def test_roundtrip_preserves_state(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.pt"
    sentinel = torch.tensor([1.0, 2.0, 3.0])
    saved = save_checkpoint({"step": 42, "model": {"w": sentinel}, "seed": 7}, path)
    assert saved == path.resolve()
    loaded = load_checkpoint(path)
    assert loaded["step"] == 42
    assert torch.equal(loaded["model"]["w"], sentinel)
    assert loaded["seed"] == 7


def test_load_restores_rng_state(tmp_path: Path) -> None:
    path = tmp_path / "rng.pt"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Draw once to advance every RNG, then snapshot.
    _ = random.random(), np.random.rand(), torch.rand(1)
    save_checkpoint({"step": 0}, path)

    # Perturb every RNG, then load and draw again — values should match the post-save draw.
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    load_checkpoint(path, restore_rng=True)
    restored = (random.random(), np.random.rand(), float(torch.rand(1)))

    # Re-run the identical sequence from scratch as the reference.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    _ = random.random(), np.random.rand(), torch.rand(1)
    expected = (random.random(), np.random.rand(), float(torch.rand(1)))

    assert restored == expected


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "ckpt.pt"
    save_checkpoint({"v": 1}, nested)
    assert nested.is_file()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    try:
        load_checkpoint(missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")  # pragma: no cover
