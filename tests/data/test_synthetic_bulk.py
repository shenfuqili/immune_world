"""Unit tests for `SyntheticBulkDataset` (Sec. 4.6 Deconvolution)."""

from __future__ import annotations

import torch

from immune_world.data.synthetic_bulk import SyntheticBulkDataset


def test_synthetic_bulk_shapes_and_simplex() -> None:
    n_cells, n_genes, n_cell_types = 80, 16, 4
    reference = torch.randn(n_cells, n_genes)
    cell_type_ids = torch.randint(0, n_cell_types, (n_cells,))
    ds = SyntheticBulkDataset(
        reference,
        cell_type_ids,
        n_cell_types=n_cell_types,
        n_mixtures=20,
        cells_per_mixture=32,
        seed=7,
    )
    assert len(ds) == 20
    item = ds[0]
    assert item["bulk"].shape == (n_genes,)
    assert item["proportions"].shape == (n_cell_types,)
    # Proportions must live on the simplex.
    assert torch.allclose(item["proportions"].sum(), torch.tensor(1.0), atol=1e-5)
    assert (item["proportions"] >= 0).all()


def test_synthetic_bulk_is_deterministic_for_fixed_seed() -> None:
    reference = torch.randn(40, 8)
    cell_type_ids = torch.randint(0, 2, (40,))

    a = SyntheticBulkDataset(reference, cell_type_ids, n_cell_types=2, n_mixtures=5, seed=42)
    b = SyntheticBulkDataset(reference, cell_type_ids, n_cell_types=2, n_mixtures=5, seed=42)
    item_a = a[0]
    item_b = b[0]
    assert torch.allclose(item_a["bulk"], item_b["bulk"])
    assert torch.allclose(item_a["proportions"], item_b["proportions"])


def test_synthetic_bulk_rejects_mismatched_inputs() -> None:
    reference = torch.randn(10, 4)
    bad_ids = torch.randint(0, 2, (5,))  # wrong length
    try:
        SyntheticBulkDataset(reference, bad_ids, n_cell_types=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched cell_type_ids")  # pragma: no cover
