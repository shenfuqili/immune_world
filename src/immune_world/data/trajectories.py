"""Trajectory windowing + scVelo benchmark dataset wrappers.

Ref: Sec. 2.1 — Pancreas (n=3,696), Dentate Gyrus (n=2,930), Bone Marrow (n=5,780).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    import anndata as ad
    from torch import Tensor


class TrajectoryDataset(Dataset[dict[str, "Tensor"]]):
    """Yields trajectory windows (cells ordered by pseudo-time) as `dict[str, Tensor]` items."""

    def __init__(self, adata: ad.AnnData, window: int = 32) -> None:
        self.adata = adata
        self.window = window

    def __len__(self) -> int:
        raise NotImplementedError("Sec. 2.3")

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raise NotImplementedError("Sec. 2.3")
