"""Pretraining corpus assembly: HCA 5.2M + CellxGene 4.8M + GEO 2.4M = 12.4M immune-enriched cells.

Ref: Sec. 4.6 — "Three separate sources have contributed to create our pre-training corpus".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torch import Tensor


class PretrainCorpus(Dataset[dict[str, "Tensor"]]):
    """Concatenates HCA + CellxGene + GEO sources into a single streaming dataset."""

    def __init__(
        self,
        hca_path: str,
        cellxgene_path: str,
        geo_path: str,
        hvgs: list[str] | None = None,
    ) -> None:
        self.hca_path = hca_path
        self.cellxgene_path = cellxgene_path
        self.geo_path = geo_path
        self.hvgs = hvgs

    def __len__(self) -> int:
        raise NotImplementedError("Sec. 4.6 Pretraining")

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raise NotImplementedError("Sec. 4.6 Pretraining")

    def source_counts(self) -> dict[str, int]:
        """Return cell counts per source (expected: ~5.2M / 4.8M / 2.4M)."""
        raise NotImplementedError("Sec. 4.6 Pretraining")
