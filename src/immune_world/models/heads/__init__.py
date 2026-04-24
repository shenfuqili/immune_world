"""Task-specific heads on top of the shared ImmuneWorld backbone (Sec. 4.6 Fine-tuning)."""

from __future__ import annotations

from immune_world.models.heads.deconvolution import DeconvolutionHead
from immune_world.models.heads.icb import ICBResponseHead
from immune_world.models.heads.perturbation import PerturbationHead
from immune_world.models.heads.trajectory import TrajectoryHead

__all__ = ["DeconvolutionHead", "ICBResponseHead", "PerturbationHead", "TrajectoryHead"]
