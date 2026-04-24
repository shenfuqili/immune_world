"""One transformer layer: TrajAttn + LayerNorm + SwiGLU FFN with residual connections.

Ref: Sec. 4.3, Eq. (3) — pre-norm residual block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from immune_world.models.swiglu_ffn import SwiGLUFFN
from immune_world.models.trajectory_attention import TrajAttention

if TYPE_CHECKING:
    from torch import Tensor


class TransformerLayer(nn.Module):
    """Eq. (3): `h^{(ℓ)} = TrajAttn(LN(h^{(ℓ-1)})) + h^{(ℓ-1)}`; `FFN(LN(·)) + ·`."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TrajAttention(d_model, n_heads, alpha_init, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_mult, dropout)

    def forward(self, h: Tensor, pseudo_time: Tensor) -> Tensor:
        h = h + self.attn(self.norm1(h), pseudo_time)
        out: Tensor = h + self.ffn(self.norm2(h))
        return out
