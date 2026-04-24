"""Smoke tests for scaffolding utilities that have real bodies (not NotImplementedError)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_set_seed_restores_reproducibility() -> None:
    import numpy as np
    import torch

    from immune_world.utils.seeding import set_seed

    set_seed(123)
    a = torch.rand(4)
    n = np.random.rand(4)
    set_seed(123)
    b = torch.rand(4)
    m = np.random.rand(4)

    assert torch.allclose(a, b)
    assert np.allclose(n, m)


def test_resolve_dataset_path_uses_override(tmp_path: Path) -> None:
    from immune_world.utils.paths import resolve_dataset_path

    resolved = resolve_dataset_path("norman", override=tmp_path)
    assert resolved == tmp_path.resolve()


def test_resolve_dataset_path_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from immune_world.utils.paths import resolve_dataset_path

    monkeypatch.setenv("IMMUNE_WORLD_DATA_DIR", str(tmp_path))
    resolved = resolve_dataset_path("norman")
    assert resolved.parent == tmp_path.resolve()


def test_registry_registers_and_looks_up() -> None:
    from immune_world.utils.registry import Registry

    registry: Registry[str] = Registry("demo")

    @registry.register("a")
    def _make_a() -> str:
        return "A"

    assert registry.get("a")() == "A"
    assert registry.keys() == ["a"]
