"""Cell-type deconvolution head — softmax-normalised linear layer over 8 cell types.

Ref: Sec. 4.6 Deconvolution — "a softmax-normalised linear layer method".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class DeconvolutionHead(nn.Module):
    """Pooled bulk embedding → cell-type proportion simplex."""

    def __init__(self, d_model: int = 512, n_cell_types: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_cell_types)

    def forward(self, h_pooled: Tensor) -> Tensor:
        """h_pooled: (B, d_model) — pooled across cells in a synthetic bulk."""
        return F.softmax(self.proj(h_pooled), dim=-1)
