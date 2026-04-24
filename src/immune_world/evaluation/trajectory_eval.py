"""Trajectory-reconstruction evaluation — Table 1 CBDir/DTW + Table S1 per-dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def evaluate_trajectory(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 1 trajectory columns")
