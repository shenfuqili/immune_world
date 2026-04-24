"""Gene embedding + continuous sinusoidal temporal positional encoding.

Ref: Sec. 4.2, Eq. (2) — `z_t = LayerNorm(x_t^T E + p_t)`, E initialised from Gene2Vec [32].
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class GeneEmbedding(nn.Module):
    """Gene-token embedding E ∈ R^{G × d} + continuous-time sinusoidal position."""

    def __init__(
        self,
        n_genes: int = 2000,
        d_model: int = 512,
        gene2vec_init: Tensor | None = None,
        max_pseudo_time: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal encoding; got {d_model}")
        self.n_genes = n_genes
        self.d_model = d_model
        self.max_pseudo_time = max_pseudo_time

        if gene2vec_init is not None:
            if gene2vec_init.shape != (n_genes, d_model):
                raise ValueError(
                    f"gene2vec_init shape {tuple(gene2vec_init.shape)} "
                    f"!= expected ({n_genes}, {d_model})"
                )
            self.E = nn.Parameter(gene2vec_init.clone())
        else:
            self.E = nn.Parameter(torch.empty(n_genes, d_model))
            nn.init.normal_(self.E, mean=0.0, std=1.0 / d_model**0.5)

        self.norm = nn.LayerNorm(d_model)

        half = d_model // 2
        # Sinusoidal frequency base follows Vaswani 2017 [31]: `10000^(2k/d)`.
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(2.0 * math.log(10000.0) / d_model)
        )
        self.register_buffer("_freqs", freqs, persistent=False)

    def forward(self, x: Tensor, pseudo_time: Tensor) -> Tensor:
        """x: (B, T, G) expression vectors; pseudo_time: (B, T) ∈ [0, max_pseudo_time]."""
        if x.shape[-1] != self.n_genes:
            raise ValueError(f"expected last dim = n_genes={self.n_genes}, got {x.shape[-1]}")
        if pseudo_time.shape != x.shape[:2]:
            raise ValueError(
                f"pseudo_time shape {tuple(pseudo_time.shape)} != x batch dims {tuple(x.shape[:2])}"
            )

        embedded = x @ self.E  # (B, T, d_model)
        positional = self._build_positional(pseudo_time)  # (B, T, d_model)
        out: Tensor = self.norm(embedded + positional)
        return out

    def _build_positional(self, pseudo_time: Tensor) -> Tensor:
        """Continuous sinusoidal encoding (paper p.27 L618–623)."""
        # pseudo_time: (B, T) ∈ [0, max_pseudo_time]; scaled to the sinusoidal base.
        scaled_time = pseudo_time.unsqueeze(-1)  # (B, T, 1)
        freqs = self._freqs.to(device=pseudo_time.device, dtype=pseudo_time.dtype)
        angles = scaled_time * freqs  # (B, T, d_model / 2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
