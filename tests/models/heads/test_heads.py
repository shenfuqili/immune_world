"""Unit tests for the four task heads (Sec. 4.6 Fine-tuning)."""

from __future__ import annotations

import torch

from immune_world.models.heads.deconvolution import DeconvolutionHead
from immune_world.models.heads.icb import ICBResponseHead
from immune_world.models.heads.perturbation import PerturbationHead
from immune_world.models.heads.trajectory import TrajectoryHead


def test_perturbation_head_output_shape() -> None:
    head = PerturbationHead(d_model=16, n_genes=32)
    h = torch.randn(2, 5, 16)
    out = head(h)
    assert out.shape == (2, 5, 32)


def test_trajectory_head_output_shape() -> None:
    head = TrajectoryHead(d_model=16, n_genes=32, hidden=64)
    h = torch.randn(2, 5, 16)
    assert head(h).shape == (2, 5, 32)


def test_deconvolution_head_produces_simplex() -> None:
    head = DeconvolutionHead(d_model=16, n_cell_types=4)
    h = torch.randn(3, 16)
    props = head(h)
    assert props.shape == (3, 4)
    assert torch.allclose(props.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert (props >= 0).all()


def test_icb_head_mean_pools_and_outputs_logit() -> None:
    head = ICBResponseHead(d_model=16, hidden=8, dropout=0.0)
    cells = torch.randn(3, 20, 16)  # 3 patients, 20 cells each
    logits = head(cells)
    assert logits.shape == (3, 1)
