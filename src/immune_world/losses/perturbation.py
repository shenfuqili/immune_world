"""Perturbation-prediction loss applied to cells with known perturbation outcomes.

Ref: Sec. 4.6 — `L_pert` "applied to cells with known perturbation outcomes".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class PerturbationPredictionLoss(nn.Module):
    """MSE between predicted post-perturbation expression and ground-truth Perturb-Seq outcomes.

    `perturbed_mask` flags the cells / genes that actually have a known post-perturbation outcome.
    The loss averages over that mask only.
    """

    def forward(self, pred: Tensor, target: Tensor, perturbed_mask: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )
        mask_f = perturbed_mask.to(pred.dtype)
        # Allow the mask to broadcast (e.g., per-cell flag over (B, T, G)). The denominator uses
        # the broadcast mask so every valid scalar position contributes one unit of "n_valid".
        while mask_f.dim() < pred.dim():
            mask_f = mask_f.unsqueeze(-1)
        mask_f = mask_f.expand_as(pred)
        if mask_f.sum() == 0:
            return pred.new_zeros(())
        squared_err = (pred - target).pow(2) * mask_f
        return squared_err.sum() / torch.clamp(mask_f.sum(), min=1.0)
