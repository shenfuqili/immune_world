"""Unit tests for the GRL + CrossCancerHead (Sec. 4.4, Eq. 6)."""

from __future__ import annotations

import torch

from immune_world.models.cross_cancer_transfer import CrossCancerHead, grl


def test_grl_forward_is_identity() -> None:
    x = torch.randn(3, 8)
    y = grl(x, lambda_adv=0.5)
    assert torch.equal(x, y)


def test_grl_negates_gradient_with_lambda() -> None:
    x = torch.randn(3, 8, requires_grad=True)
    y = grl(x, lambda_adv=0.1)
    y.sum().backward()
    assert x.grad is not None
    # Gradient = -λ * dL/dy = -λ * 1 (since y.sum() d/dy = 1s). So every grad entry = -0.1.
    assert torch.allclose(x.grad, torch.full_like(x.grad, -0.1))


def test_cross_cancer_head_output_shapes() -> None:
    head = CrossCancerHead(d_model=32, hidden=16, n_cancer_types=4, lambda_adv=0.1)
    h = torch.randn(5, 32)
    shared, logits = head(h)
    assert shared.shape[0] == 5 and shared.shape[-1] == max(1, 32 // 4)
    assert logits.shape == (5, 4)
