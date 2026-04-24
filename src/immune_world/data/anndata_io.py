"""AnnData / HDF5 loaders for scRNA-seq and bulk RNA-seq inputs.

Ref: Sec. 4.6 — "Three separate sources have contributed to create our pre-training corpus"
(Human Cell Atlas / CellxGene / GEO).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad


def load_anndata(path: str | Path) -> ad.AnnData:
    """Load a .h5ad file; fail loudly if the file is missing or not an AnnData object."""
    raise NotImplementedError("Sec. 4.6")


def write_anndata(adata: ad.AnnData, path: str | Path) -> None:
    """Write AnnData to .h5ad atomically (tmp file + rename)."""
    raise NotImplementedError("Sec. 4.6")
