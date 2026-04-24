"""Unit tests for the composite training objective and its three branches (Sec. 4.6, Eq. 9)."""

from __future__ import annotations

import torch

from immune_world.losses.composite import CompositeObjective
from immune_world.losses.cross_cancer import CrossCancerAdversarialLoss
from immune_world.losses.perturbation import PerturbationPredictionLoss
from immune_world.losses.reconstruction import MaskedGeneBCELoss
from immune_world.losses.trajectory import TrajectoryMSELoss


def test_trajectory_mse_on_identical_returns_zero() -> None:
    loss = TrajectoryMSELoss()
    x = torch.randn(3, 5, 8)
    assert float(loss(x, x)) == 0.0


def test_trajectory_mse_rejects_shape_mismatch() -> None:
    loss = TrajectoryMSELoss()
    try:
        loss(torch.randn(3, 5, 8), torch.randn(3, 5, 4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")  # pragma: no cover


def test_masked_bce_ignores_unmasked_positions() -> None:
    loss_fn = MaskedGeneBCELoss()
    pred = torch.zeros(2, 4)  # logits of 0 → sigmoid 0.5 → BCE ~0.693
    target = torch.ones(2, 4)  # all ones
    mask_none = torch.zeros(2, 4, dtype=torch.bool)
    mask_all = torch.ones(2, 4, dtype=torch.bool)
    assert float(loss_fn(pred, target, mask_none)) == 0.0
    non_zero = float(loss_fn(pred, target, mask_all))
    assert non_zero > 0.5


def test_perturbation_loss_honours_mask() -> None:
    loss_fn = PerturbationPredictionLoss()
    pred = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    no_mask = torch.zeros(2, dtype=torch.bool)
    assert float(loss_fn(pred, target, no_mask)) == 0.0
    all_mask = torch.ones(2, dtype=torch.bool)
    assert float(loss_fn(pred, target, all_mask)) == 1.0  # MSE = 1^2 = 1


def test_composite_weights_match_paper() -> None:
    obj = CompositeObjective(lambda_recon=0.5, lambda_pert=0.3)
    assert obj.lambda_recon == 0.5
    assert obj.lambda_pert == 0.3


def test_composite_forward_returns_expected_schema() -> None:
    obj = CompositeObjective(lambda_recon=0.5, lambda_pert=0.3)
    x = torch.randn(1, 3, 5)
    y = x + 0.1
    mask = torch.ones_like(x, dtype=torch.bool)
    out = obj(x, y, x, x.clamp(0, 1), mask, x, y, torch.ones(1, dtype=torch.bool))
    assert set(out.keys()) == {"total", "traj", "recon", "pert"}
    # total == traj + 0.5*recon + 0.3*pert
    manual = out["traj"] + 0.5 * out["recon"] + 0.3 * out["pert"]
    assert torch.allclose(out["total"], manual, atol=1e-6)


def test_cross_cancer_loss_is_cross_entropy() -> None:
    loss_fn = CrossCancerAdversarialLoss(lambda_adv=0.1)
    logits = torch.zeros(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    # All logits equal → uniform softmax → loss = log(3).
    val = float(loss_fn(logits, labels))
    assert abs(val - float(torch.log(torch.tensor(3.0)))) < 1e-5
