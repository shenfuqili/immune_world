"""Perturbation-prediction evaluation — Table 1 perturbation columns + Table S1 extended metrics.

Ref: Table 1 (Pearson r per dataset) + Table S1 (precision / recall / specificity / MAE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def evaluate_perturbation(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 1 perturbation columns")
