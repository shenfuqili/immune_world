"""Unit tests for `GeneEmbedding` (Sec. 4.2, Eq. 2)."""

from __future__ import annotations

import torch

from immune_world.models.gene_embedding import GeneEmbedding


def test_gene_embedding_output_shape() -> None:
    embed = GeneEmbedding(n_genes=64, d_model=32)
    x = torch.randn(4, 8, 64)
    t = torch.linspace(0, 1, 8).unsqueeze(0).expand(4, -1)
    z = embed(x, t)
    assert z.shape == (4, 8, 32)


def test_gene_embedding_rejects_wrong_gene_dim() -> None:
    embed = GeneEmbedding(n_genes=64, d_model=32)
    x = torch.randn(2, 4, 32)
    t = torch.zeros(2, 4)
    try:
        embed(x, t)
    except ValueError as exc:
        assert "n_genes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_sinusoidal_positional_is_pairwise_distinct() -> None:
    embed = GeneEmbedding(n_genes=16, d_model=8)
    t = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    pe = embed._build_positional(t)
    # Different time steps should not collapse to the same encoding.
    for a in range(pe.shape[1]):
        for b in range(a + 1, pe.shape[1]):
            assert not torch.allclose(pe[0, a], pe[0, b], atol=1e-6)


def test_gene2vec_init_is_honoured() -> None:
    init = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
    embed = GeneEmbedding(n_genes=64, d_model=32, gene2vec_init=init)
    assert torch.allclose(embed.E.data, init)
