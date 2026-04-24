"""Perturb-Seq datasets (Norman / Adamson / Replogle K562 / RPE1) + perturbation-identity split.

Ref: Sec. 2.1 — 80/10/10 split "based on perturbation identity, ensuring that no perturbation was
present in both the training and testing datasets".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    import anndata as ad
    from torch import Tensor


class PerturbSeqDataset(Dataset[dict[str, "Tensor"]]):
    """Iterates (cell, perturbation, post-pert expression) triples."""

    def __init__(self, adata: ad.AnnData, *, perturbation_col: str = "perturbation") -> None:
        self.adata = adata
        self.perturbation_col = perturbation_col

    def __len__(self) -> int:
        raise NotImplementedError("Sec. 2.2")

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raise NotImplementedError("Sec. 2.2")


def perturbation_identity_split(
    adata: ad.AnnData,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Return (train, val, test) AnnData whose perturbation sets are pairwise disjoint."""
    raise NotImplementedError("Sec. 2.1")
