"""Two-way component-interaction analysis — Table S2 Δ_obs − Δ_exp synergy columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def compute_two_way_interactions(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Each pair drop → {Value, Δ_obs, Interaction} per (Norman, CBDir, ICB AUC)."""
    raise NotImplementedError("Table S2")
