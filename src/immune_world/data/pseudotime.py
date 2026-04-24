"""Diffusion-pseudotime inference + entropy-based filtering.

Ref: Sec. 4.6 — "we determined pseudotimepoints through calculating diffusion pseudotime [30]
based on the 50 most significant components in the PCA representation ... Cells that are ambiguous
(are assigned diffusion pseudotime scores greater than 0.9) will be eliminated".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


def diffusion_pseudotime(
    adata: ad.AnnData,
    n_pcs: int = 50,
    entropy_cut: float = 0.9,
) -> ad.AnnData:
    """Annotate `adata.obs["pseudotime"]` and drop ambiguous cells (DPT entropy > `entropy_cut`)."""
    raise NotImplementedError("Sec. 4.6, ref [30] Haghverdi et al.")
