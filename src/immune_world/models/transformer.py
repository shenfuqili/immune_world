"""12-layer trajectory-aware transformer stack with causal masking.

Ref: Sec. 4.3 — `L = 12` layers, d_model = 512, H = 8 heads, d_k = 64; Table S3 defaults (†).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from immune_world.models.transformer_layer import TransformerLayer

if TYPE_CHECKING:
    from torch import Tensor


class TrajectoryAwareTransformer(nn.Module):
    """Stack of `n_layers` TransformerLayers; output is the contextualised cell-state sequence."""

    def __init__(
        self,
        n_layers: int = 12,
        d_model: int = 512,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, ffn_mult, dropout, alpha_init)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, pseudo_time: Tensor) -> Tensor:
        for layer in self.layers:
            h = layer(h, pseudo_time)
        out: Tensor = self.final_norm(h)
        return out
