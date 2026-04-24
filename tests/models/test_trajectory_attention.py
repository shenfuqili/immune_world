"""Unit tests for `TrajAttention` (Sec. 4.3, Eq. 4)."""

from __future__ import annotations

import torch

from immune_world.models.trajectory_attention import TrajAttention


def test_trajectory_attention_output_shape() -> None:
    attn = TrajAttention(d_model=32, n_heads=4, alpha_init=0.01)
    h = torch.randn(2, 6, 32)
    t = torch.linspace(0, 1, 6).unsqueeze(0).expand(2, -1)
    y = attn(h, t)
    assert y.shape == h.shape


def test_alpha_is_per_head_parameter() -> None:
    attn = TrajAttention(d_model=32, n_heads=8, alpha_init=0.05)
    assert attn.alpha.shape == (8,)
    assert torch.allclose(attn.alpha, torch.full((8,), 0.05))
    # Parameter must require grad so the paper's claim of *learnable* decay holds.
    assert attn.alpha.requires_grad


def test_causal_mask_prevents_future_leakage() -> None:
    """Output at step t must be independent of input at steps > t."""
    torch.manual_seed(0)
    attn = TrajAttention(d_model=16, n_heads=2, alpha_init=0.0)  # α=0 disables temporal bias
    attn.eval()

    h = torch.randn(1, 4, 16)
    t = torch.linspace(0, 1, 4).unsqueeze(0)
    y_original = attn(h, t)

    # Perturb only the final time step; earlier outputs must remain unchanged under causal masking.
    h2 = h.clone()
    h2[0, -1] = torch.randn(16)
    y_perturbed = attn(h2, t)

    assert torch.allclose(y_original[0, :3], y_perturbed[0, :3], atol=1e-5)
    assert not torch.allclose(y_original[0, 3], y_perturbed[0, 3], atol=1e-5)


def test_temporal_bias_penalises_distant_cells() -> None:
    """With large α, the attention logits at distant times should be strongly negative."""
    attn = TrajAttention(d_model=8, n_heads=1, alpha_init=5.0)
    t = torch.tensor([[0.0, 1.0]])
    bias = attn._temporal_bias(t)
    # Off-diagonal entries encode |t_s - t_t| = 1 → bias = −α · 1 = −5.
    assert torch.allclose(bias[0, 0, 0, 1], torch.tensor(-5.0))
    # Diagonal entries (|Δt| = 0) → bias = 0.
    assert torch.allclose(bias[0, 0, 0, 0], torch.tensor(0.0))
