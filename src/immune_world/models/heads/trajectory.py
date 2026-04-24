"""Trajectory reconstruction head — predicts next-state gene expression.

Ref: Sec. 4.6 — "(ii) the trajectory-aware decoder used for reconstructing trajectories".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class TrajectoryHead(nn.Module):
    """Two-layer projection producing `x̂_{t+1}`. Ref: Sec. 4.6 + Eq. (1)."""

    def __init__(self, d_model: int = 512, n_genes: int = 2000, hidden: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, h: Tensor) -> Tensor:
        out: Tensor = self.mlp(h)
        return out
