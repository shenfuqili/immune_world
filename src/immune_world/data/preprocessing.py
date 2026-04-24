"""HVG selection + normalisation + log1p for the 12.4 M-cell pretraining corpus.

Ref: Sec. 4.6 — "top 2,000 most variable genes identified across all three input sources to form
our gene vocabulary (G = 2000)".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


def select_hvgs(adata: ad.AnnData, n_top: int = 2000) -> ad.AnnData:
    """Return a new AnnData filtered to the top-`n_top` highly-variable genes."""
    raise NotImplementedError("Sec. 4.6")


def normalise_and_log1p(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """Per-cell total-count normalisation followed by natural-log(1+x)."""
    raise NotImplementedError("Sec. 4.6")
