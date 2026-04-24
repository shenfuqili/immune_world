"""Cross-cancer transfer module with gradient-reversal-based adversarial domain-invariance.

Ref: Sec. 4.4, Eq. (6) — `L_transfer = L_task − λ_adv · L_disc`, λ_adv = 0.1.
GRL formulation from Ganin et al. 2016 (ref [34]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class GradientReversalFn(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -lambda on the backward pass."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_adv: float) -> Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.lambda_adv, None


def grl(x: Tensor, lambda_adv: float) -> Tensor:
    """Short alias for `GradientReversalFn.apply`."""
    # torch.autograd.Function.apply is untyped upstream; we know the return is a Tensor.
    return GradientReversalFn.apply(x, lambda_adv)  # type: ignore[no-untyped-call,no-any-return]


class CrossCancerHead(nn.Module):
    """Two-layer MLP projection `g_φ` + cancer-type discriminator (trained adversarially)."""

    def __init__(
        self,
        d_model: int = 512,
        hidden: int = 256,
        n_cancer_types: int = 7,
        lambda_adv: float = 0.1,
    ) -> None:
        super().__init__()
        shared_dim = max(1, d_model // 4)
        self.projection = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, shared_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(shared_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_cancer_types),
        )
        self.lambda_adv = lambda_adv

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Return (shared_rep, cancer_type_logits).

        The discriminator reads `grl(shared_rep)` so the encoder learns cancer-invariant features
        while the discriminator itself tries to distinguish cancer types (min-max).
        """
        shared = self.projection(h)
        reversed_shared = grl(shared, self.lambda_adv)
        cancer_logits = self.discriminator(reversed_shared)
        return shared, cancer_logits
