"""Composite three-branch training objective.

Ref: Sec. 4.6, Eq. (9) — `L_total = L_traj + λ_1 L_recon + λ_2 L_pert`, λ_1 = 0.5, λ_2 = 0.3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from torch import nn

from immune_world.losses.perturbation import PerturbationPredictionLoss
from immune_world.losses.reconstruction import MaskedGeneBCELoss
from immune_world.losses.trajectory import TrajectoryMSELoss

if TYPE_CHECKING:
    from torch import Tensor


class CompositeLossOutput(TypedDict):
    total: Tensor
    traj: Tensor
    recon: Tensor
    pert: Tensor


class CompositeObjective(nn.Module):
    """Weighted sum of the three pretraining branches (Eq. 9)."""

    def __init__(self, lambda_recon: float = 0.5, lambda_pert: float = 0.3) -> None:
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_pert = lambda_pert
        self.loss_traj = TrajectoryMSELoss()
        self.loss_recon = MaskedGeneBCELoss()
        self.loss_pert = PerturbationPredictionLoss()

    def forward(
        self,
        pred_next: Tensor,
        target_next: Tensor,
        recon_pred: Tensor,
        recon_target: Tensor,
        recon_mask: Tensor,
        pert_pred: Tensor,
        pert_target: Tensor,
        pert_mask: Tensor,
    ) -> CompositeLossOutput:
        l_traj = self.loss_traj(pred_next, target_next)
        l_recon = self.loss_recon(recon_pred, recon_target, recon_mask)
        l_pert = self.loss_pert(pert_pred, pert_target, pert_mask)
        total = l_traj + self.lambda_recon * l_recon + self.lambda_pert * l_pert
        return CompositeLossOutput(total=total, traj=l_traj, recon=l_recon, pert=l_pert)
