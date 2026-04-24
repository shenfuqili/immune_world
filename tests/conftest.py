"""Shared pytest fixtures — small synthetic datasets keep CI CPU-only and under 30 seconds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def tiny_config() -> dict[str, int | float]:
    """Config used by smoke tests — matches `configs/experiment/_smoke.yaml`."""
    return {
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "n_genes": 64,
        "batch_size": 4,
        "trajectory_len": 8,
        "lr": 1e-3,
    }


@pytest.fixture(autouse=True)
def _seed_every_test() -> Iterator[None]:
    """Seed the RNG before every test so failures are reproducible."""
    try:
        from immune_world.utils.seeding import set_seed
    except ImportError:  # pragma: no cover - package not importable yet
        yield
        return
    set_seed(42)
    yield
