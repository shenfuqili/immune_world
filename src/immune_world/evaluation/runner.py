"""Top-level evaluation dispatcher — emits the full metrics JSON that Tables 1/2 are built from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run_full_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Run every task's evaluation and return a dict whose keys match paper Tables 1/2 columns."""
    raise NotImplementedError("Table 1 / 2 runner")
