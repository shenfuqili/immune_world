"""Next-state MSE loss.

Ref: Sec. 4.6 — `L_traj = (1/(T-1)) Σ ‖x̂_{t+1} − x_{t+1}‖^2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class TrajectoryMSELoss(nn.Module):
    """MSE between predicted `x̂_{t+1}` and ground-truth `x_{t+1}` averaged over T−1 steps."""

    def forward(self, pred_next: Tensor, target_next: Tensor) -> Tensor:
        if pred_next.shape != target_next.shape:
            raise ValueError(
                f"shape mismatch: pred {tuple(pred_next.shape)} "
                f"target {tuple(target_next.shape)}"
            )
        loss: Tensor = F.mse_loss(pred_next, target_next, reduction="mean")
        return loss
