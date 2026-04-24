"""Loss branches used by the composite Eq. (9) training objective."""

from __future__ import annotations

from immune_world.losses.composite import CompositeObjective
from immune_world.losses.cross_cancer import CrossCancerAdversarialLoss
from immune_world.losses.perturbation import PerturbationPredictionLoss
from immune_world.losses.reconstruction import MaskedGeneBCELoss
from immune_world.losses.trajectory import TrajectoryMSELoss

__all__ = [
    "CompositeObjective",
    "CrossCancerAdversarialLoss",
    "MaskedGeneBCELoss",
    "PerturbationPredictionLoss",
    "TrajectoryMSELoss",
]
