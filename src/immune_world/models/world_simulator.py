"""Top-level ImmuneWorld — autoregressive immune-trajectory world simulator.

Ref: Sec. 4.1, Eq. (1) — `x̂_{t+1} = f_θ(x_{≤t}, c)`. With `c = ∅` the model performs trajectory
prediction; with `c ≠ ∅` it performs counterfactual perturbation simulation. Four task heads
(perturbation / trajectory / deconvolution / ICB response) share the same backbone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from torch import nn

from immune_world.models.cross_cancer_transfer import CrossCancerHead
from immune_world.models.gene_embedding import GeneEmbedding
from immune_world.models.heads.deconvolution import DeconvolutionHead
from immune_world.models.heads.icb import ICBResponseHead
from immune_world.models.heads.perturbation import PerturbationHead
from immune_world.models.heads.trajectory import TrajectoryHead
from immune_world.models.perturbation_engine import PerturbationEmbedder, PerturbationInjection
from immune_world.models.transformer import TrajectoryAwareTransformer

if TYPE_CHECKING:
    from torch import Tensor


class ImmuneWorldOutput(TypedDict):
    hidden: Tensor
    perturbation_pred: Tensor
    trajectory_pred: Tensor
    shared_repr: Tensor
    cancer_logits: Tensor


class ImmuneWorld(nn.Module):
    """Complete architecture: Gene embedding → 12-layer TrajAttn → cross-cancer transfer → heads."""

    def __init__(
        self,
        n_genes: int = 2000,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        n_cancer_types: int = 7,
        n_cell_types: int = 8,
        lambda_adv: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = GeneEmbedding(n_genes=n_genes, d_model=d_model)
        self.transformer = TrajectoryAwareTransformer(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.perturbation_embedder = PerturbationEmbedder(self.embed)
        self.perturbation_injector = PerturbationInjection(d_model, n_heads, dropout=dropout)
        self.cross_cancer = CrossCancerHead(
            d_model=d_model, n_cancer_types=n_cancer_types, lambda_adv=lambda_adv
        )
        self.head_perturbation = PerturbationHead(d_model, n_genes)
        self.head_trajectory = TrajectoryHead(d_model, n_genes)
        self.head_deconvolution = DeconvolutionHead(d_model, n_cell_types)
        self.head_icb = ICBResponseHead(d_model)

    def forward(
        self,
        x: Tensor,
        pseudo_time: Tensor,
        perturbation: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> ImmuneWorldOutput:
        """Ref: Sec. 4.1, Eq. (1). Returns a dict of per-head outputs.

        Each returned tensor is shape (B, T, *) except `shared_repr` / `cancer_logits` which are
        computed from the final-step hidden state (B, *).
        """
        h = self.embed(x, pseudo_time)  # (B, T, d_model)
        h = self.transformer(h, pseudo_time)

        if perturbation is not None:
            pert_genes, doses, types = perturbation
            c = self.perturbation_embedder(pert_genes, doses, types)
            h = self.perturbation_injector(h, c)

        last_step = h[:, -1]  # (B, d_model)
        shared, cancer_logits = self.cross_cancer(last_step)

        return ImmuneWorldOutput(
            hidden=h,
            perturbation_pred=self.head_perturbation(h),
            trajectory_pred=self.head_trajectory(h),
            shared_repr=shared,
            cancer_logits=cancer_logits,
        )
