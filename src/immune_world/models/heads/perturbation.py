"""Perturbation prediction head — linear projection `W_pert ∈ R^{d × G}`.

Ref: Sec. 4.6 — "(i) a linear projection (i.e., a weight matrix) for perturbation prediction".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class PerturbationHead(nn.Module):
    """Maps the contextualised cell state to per-gene expression predictions."""

    def __init__(self, d_model: int = 512, n_genes: int = 2000) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_genes, bias=True)

    def forward(self, h: Tensor) -> Tensor:
        out: Tensor = self.proj(h)
        return out
