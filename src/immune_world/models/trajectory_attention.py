"""Trajectory-aware multi-head attention with learnable per-head temporal-decay bias.

Ref: Sec. 4.3, Eq. (4) — `TrajAttn(Q,K,V) = softmax(QKᵀ/√d_k + B_temp) V`,
`B_temp[s,t] = −α·|t_s − t_t|` with α learnable per head (H=8, init 0.01 per Table S3 †).

Uses FlashAttention-2 if `flash_attn` is importable (Sec. 2.8 p.18 L433); otherwise falls back to
`torch.nn.functional.scaled_dot_product_attention` with an explicit additive bias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class TrajAttention(nn.Module):
    """Eq. (4). Per-head α is learnable; causal mask restricts attention to `≤ t`."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        alpha_init: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.full((n_heads,), alpha_init))
        self.dropout_p = dropout

    def forward(
        self,
        h: Tensor,
        pseudo_time: Tensor,
        *,
        causal: bool = True,
    ) -> Tensor:
        """h: (B, T, d_model); pseudo_time: (B, T)."""
        batch_size, seq_len, _ = h.shape
        if pseudo_time.shape != (batch_size, seq_len):
            raise ValueError(
                f"pseudo_time shape {tuple(pseudo_time.shape)} "
                f"!= expected ({batch_size}, {seq_len})"
            )

        q = self._reshape_heads(self.q_proj(h))  # (B, H, T, d_k)
        k = self._reshape_heads(self.k_proj(h))
        v = self._reshape_heads(self.v_proj(h))

        bias = self._temporal_bias(pseudo_time)  # (B, H, T, T)
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=h.device), diagonal=1
            )
            # Broadcast to (1, 1, T, T); mask positions get −∞ so softmax → 0.
            bias = bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        projected: Tensor = self.out_proj(out)
        return projected

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def _temporal_bias(self, pseudo_time: Tensor) -> Tensor:
        """Build `B_temp[b,h,s,t] = −α_h · |t_s − t_t|` ∈ (B, H, T, T)."""
        # |t_s − t_t|: (B, T, T)
        dt = (pseudo_time.unsqueeze(-1) - pseudo_time.unsqueeze(-2)).abs()
        # α: (H,) → (1, H, 1, 1); dt: (B, 1, T, T)
        return -(self.alpha.view(1, self.n_heads, 1, 1) * dt.unsqueeze(1))
