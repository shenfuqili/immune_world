"""Extra task-specific baselines — Table S7.

Wraps Monocle3 / Slingshot / RNA-ODE / MuSiC / BisqueRNA / SCENIC+ / Linear / MLP / TIDE /
PD-L1 / TMB where upstream libraries exist; analytical floors are used where they don't.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def evaluate_extra_baselines(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Keyed by baseline name → metrics dict (Pearson r, MAE, CBDir, DTW, F1, AUC)."""
    raise NotImplementedError("Table S7")
