"""Immunotherapy response head — 2-layer MLP with dropout 0.1 over mean-pooled patient embedding.

Ref: Sec. 4.6 Immunotherapy response prediction — "mean pooling ImmuneWorld cell embeddings ...
2-layer MLP with dropout = 0.1 ... AUC described as our primary metric".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class ICBResponseHead(nn.Module):
    """Patient-level aggregation → binary responder classifier (CR+PR vs SD+PD)."""

    def __init__(self, d_model: int = 512, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cell_embeddings: Tensor) -> Tensor:
        """cell_embeddings: (B_patients, N_cells, d_model) → mean-pool → logit (B, 1)."""
        pooled = cell_embeddings.mean(dim=1)
        out: Tensor = self.mlp(pooled)
        return out
