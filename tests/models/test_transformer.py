"""Unit tests for the TransformerLayer stack (Sec. 4.3)."""

from __future__ import annotations

import torch

from immune_world.models.transformer import TrajectoryAwareTransformer
from immune_world.models.transformer_layer import TransformerLayer


def test_single_layer_output_shape() -> None:
    layer = TransformerLayer(d_model=32, n_heads=4)
    h = torch.randn(2, 4, 32)
    t = torch.linspace(0, 1, 4).unsqueeze(0).expand(2, -1)
    y = layer(h, t)
    assert y.shape == h.shape


def test_full_stack_output_shape() -> None:
    model = TrajectoryAwareTransformer(n_layers=3, d_model=32, n_heads=4)
    h = torch.randn(2, 6, 32)
    t = torch.linspace(0, 1, 6).unsqueeze(0).expand(2, -1)
    y = model(h, t)
    assert y.shape == h.shape


def test_stack_is_differentiable() -> None:
    model = TrajectoryAwareTransformer(n_layers=2, d_model=16, n_heads=2)
    h = torch.randn(1, 3, 16, requires_grad=True)
    t = torch.zeros(1, 3)
    loss = model(h, t).sum()
    loss.backward()
    assert h.grad is not None and h.grad.shape == h.shape
