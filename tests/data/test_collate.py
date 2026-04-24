"""Unit tests for `collate_trajectory_batch` and `apply_gene_mlm_mask`."""

from __future__ import annotations

import torch

from immune_world.data.collate import apply_gene_mlm_mask, collate_trajectory_batch


def test_collate_right_pads_variable_trajectories() -> None:
    batch = [
        {"x": torch.randn(3, 8), "pseudo_time": torch.linspace(0, 0.3, 3)},
        {"x": torch.randn(5, 8), "pseudo_time": torch.linspace(0, 0.5, 5)},
    ]
    out = collate_trajectory_batch(batch)
    assert out["x"].shape == (2, 5, 8)
    assert out["pseudo_time"].shape == (2, 5)
    assert out["valid_mask"].shape == (2, 5)
    # First trajectory has 3 real + 2 padded time steps.
    assert out["valid_mask"][0].tolist() == [True, True, True, False, False]
    assert out["valid_mask"][1].all()


def test_collate_carries_perturbation_fields() -> None:
    batch = [
        {
            "x": torch.randn(2, 4),
            "pseudo_time": torch.tensor([0.0, 0.5]),
            "perturbation_genes": torch.tensor([1, 2]),
            "doses": torch.tensor([1.0]),
            "types": torch.tensor(0),
        },
        {
            "x": torch.randn(2, 4),
            "pseudo_time": torch.tensor([0.1, 0.6]),
            "perturbation_genes": torch.tensor([3]),
            "doses": torch.tensor([0.5]),
            "types": torch.tensor(1),
        },
    ]
    out = collate_trajectory_batch(batch)
    assert out["perturbation_genes"].shape == (2, 2)
    # Second sample had only 1 pert gene; padding position should be -1.
    assert int(out["perturbation_genes"][1, 1]) == -1
    assert out["doses"].shape == (2, 1)
    assert out["types"].shape == (2,)


def test_collate_rejects_empty_batch() -> None:
    try:
        collate_trajectory_batch([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty batch")  # pragma: no cover


def test_apply_gene_mlm_mask_respects_rate_bounds() -> None:
    x = torch.randn(100, 50)
    _, mask = apply_gene_mlm_mask(x, rate=0.0)
    assert mask.sum() == 0
    _, mask_all = apply_gene_mlm_mask(x, rate=1.0)
    assert mask_all.all()


def test_apply_gene_mlm_mask_is_deterministic_with_generator() -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 8)
    _, m1 = apply_gene_mlm_mask(x, rate=0.3, generator=gen)
    gen = torch.Generator().manual_seed(0)
    _, m2 = apply_gene_mlm_mask(x, rate=0.3, generator=gen)
    assert torch.equal(m1, m2)
