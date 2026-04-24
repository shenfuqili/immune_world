"""Unit tests for `immune_world.models.swiglu_ffn.SwiGLUFFN` (Sec. 4.3, Eq. 5)."""

from __future__ import annotations

import torch

from immune_world.models.swiglu_ffn import SwiGLUFFN


def test_swiglu_ffn_preserves_shape() -> None:
    ffn = SwiGLUFFN(d_model=32, ffn_mult=4)
    x = torch.randn(2, 5, 32)
    y = ffn(x)
    assert y.shape == x.shape


def test_swiglu_ffn_is_differentiable() -> None:
    ffn = SwiGLUFFN(d_model=16, ffn_mult=2)
    x = torch.randn(3, 4, 16, requires_grad=True)
    loss = ffn(x).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
