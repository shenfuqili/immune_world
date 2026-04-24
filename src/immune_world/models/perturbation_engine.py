"""Perturbation context aggregation + cross-attention injection into the transformer stack.

Ref: Sec. 4.5, Eq. (7) — `c = (1/k) Σ e_{g_j} + W_c [dose_j; type_j]`.
Eq. (8) — `h̃^{(ℓ)} = CrossAttn(h^{(ℓ)}, c) + h^{(ℓ)}` (query = cell state; key/value = c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from immune_world.models.gene_embedding import GeneEmbedding


class PerturbationEmbedder(nn.Module):
    """Mean-pools gene-specific embeddings + modulates with dose & type tags. Eq. (7)."""

    def __init__(
        self,
        gene_embedding: GeneEmbedding,  # shared GeneEmbedding (not re-initialised)
        n_perturbation_types: int = 2,  # {CRISPR-KO, overexpression}
        dose_dim: int = 1,
    ) -> None:
        super().__init__()
        self.gene_embedding = gene_embedding
        self.n_perturbation_types = n_perturbation_types
        self.dose_dim = dose_dim
        self.W_c = nn.Linear(dose_dim + n_perturbation_types, gene_embedding.d_model)

    def forward(self, perturbation_genes: Tensor, doses: Tensor, types: Tensor) -> Tensor:
        """Return c ∈ R^{B × d_model}.

        perturbation_genes: (B, k) — integer gene indices; value -1 marks "no perturbation" slots
            (padding for variable-k batches). Handled by a mean over the *non-padded* entries.
        doses: (B, dose_dim) — normalised dosage.
        types: (B,) — perturbation type index in [0, n_perturbation_types).
        """
        gene_mask = (perturbation_genes >= 0).float()  # (B, k)
        safe_idx = perturbation_genes.clamp(min=0)
        # Shared E: (G, d_model). Lookup → (B, k, d_model).
        e_genes = self.gene_embedding.E[safe_idx]
        weighted = e_genes * gene_mask.unsqueeze(-1)
        n_valid = gene_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = weighted.sum(dim=1) / n_valid  # (B, d_model)

        type_oh = F.one_hot(types, num_classes=self.n_perturbation_types).float()
        modulator = self.W_c(torch.cat([doses, type_oh], dim=-1))  # (B, d_model)
        context: Tensor = pooled + modulator
        return context


class PerturbationInjection(nn.Module):
    """Cross-attention injector. Eq. (8) — adds to layer ℓ output along residual stream."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """h: (B, T, d_model); c: (B, d_model) — single-token perturbation context per batch."""
        kv = c.unsqueeze(1)  # (B, 1, d_model)
        out, _ = self.cross_attn(h, kv, kv, need_weights=False)  # (B, T, d_model)
        injected: Tensor = h + out
        return injected
