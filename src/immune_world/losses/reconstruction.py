"""Masked-gene binary-cross-entropy reconstruction loss (MLM-style).

Ref: Sec. 4.6 — `L_recon = −(1/G) Σ [x_g log x̂_g + (1-x_g) log(1-x̂_g)]` applied to masked genes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class MaskedGeneBCELoss(nn.Module):
    """Binary-cross-entropy on masked gene positions only; non-masked positions are ignored."""

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(pred.shape)} "
                f"target {tuple(target.shape)} mask {tuple(mask.shape)}"
            )
        if mask.sum() == 0:
            return pred.new_zeros(())
        # Treat expression as Bernoulli probability; input is assumed to be in [0, 1].
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        mask_f = mask.to(bce.dtype)
        return (bce * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)
