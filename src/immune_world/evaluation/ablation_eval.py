"""Ablation-study driver — Table 3 (7 component drops + reduced corpus).

Ref: Sec. 2.6 Ablation Study, Table 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run_ablation_matrix(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Return nested dict indexed by variant name → metrics dict, matching Table 3 rows."""
    raise NotImplementedError("Table 3")
