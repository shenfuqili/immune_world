"""Unit tests for the PerturbationEmbedder and PerturbationInjection (Sec. 4.5, Eq. 7-8)."""

from __future__ import annotations

import torch

from immune_world.models.gene_embedding import GeneEmbedding
from immune_world.models.perturbation_engine import PerturbationEmbedder, PerturbationInjection


def test_perturbation_embedder_output_shape() -> None:
    gene_embed = GeneEmbedding(n_genes=20, d_model=16)
    embedder = PerturbationEmbedder(gene_embed, n_perturbation_types=2, dose_dim=1)
    pert_genes = torch.tensor([[0, 3, -1], [5, 7, 9]])  # 2 batches, k=3 (with -1 padding)
    doses = torch.tensor([[1.0], [0.5]])
    types = torch.tensor([0, 1])
    c = embedder(pert_genes, doses, types)
    assert c.shape == (2, 16)


def test_perturbation_injection_adds_to_residual() -> None:
    injector = PerturbationInjection(d_model=16, n_heads=2)
    h = torch.randn(2, 4, 16)
    c = torch.randn(2, 16)
    out = injector(h, c)
    assert out.shape == h.shape
    # Must not be identical to input (cross-attention contribution is non-trivial).
    assert not torch.allclose(out, h, atol=1e-5)


def test_padded_genes_contribute_zero_to_pooled_context() -> None:
    gene_embed = GeneEmbedding(n_genes=10, d_model=8)
    embedder = PerturbationEmbedder(gene_embed, n_perturbation_types=2, dose_dim=1)
    # Compare: one batch with padding (-1) vs. the same without padding; pooled means should match.
    doses = torch.tensor([[0.0]])
    types = torch.tensor([0])
    with_padding = embedder(torch.tensor([[0, 1, -1]]), doses, types)
    no_padding = embedder(torch.tensor([[0, 1]]), doses, types)
    assert torch.allclose(with_padding, no_padding, atol=1e-6)
