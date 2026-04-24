"""ICB cohort bulk RNA-seq dataset + leave-one-cohort-out cross-validation split.

Ref: Sec. 2.1 — 14 cohorts across 7 cancer types, compiled via ICBatlas (Tang et al. 2023 [24]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterator

    import anndata as ad
    from torch import Tensor


class ICBDataset(Dataset[dict[str, "Tensor"]]):
    """Patient-level bulk RNA-seq with binary response labels (CR+PR vs SD+PD)."""

    def __init__(self, adata: ad.AnnData, *, cohort_col: str = "cohort") -> None:
        self.adata = adata
        self.cohort_col = cohort_col

    def __len__(self) -> int:
        raise NotImplementedError("Sec. 2.5")

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raise NotImplementedError("Sec. 2.5")


def leave_one_cohort_out_split(
    adata: ad.AnnData, *, cohort_col: str = "cohort"
) -> Iterator[tuple[ad.AnnData, ad.AnnData]]:
    """Yield (train_all_but_k, test_k) splits across all unique cohorts."""
    raise NotImplementedError("Sec. 2.5")
