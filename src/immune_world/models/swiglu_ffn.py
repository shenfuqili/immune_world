"""SwiGLU feed-forward network.

Ref: Sec. 4.3, Eq. (5) — `FFN(h) = (W_1 h ⊙ SiLU(W_g h)) W_2`, `W_{1,g} ∈ R^{d×4d}` (Shazeer 2020).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class SwiGLUFFN(nn.Module):
    """Shazeer-style gated-linear FFN using SiLU/Swish activation."""

    def __init__(self, d_model: int = 512, ffn_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        d_hidden = d_model * ffn_mult
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.wg = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        out: Tensor = self.w2(self.dropout(self.w1(h) * F.silu(self.wg(h))))
        return out
