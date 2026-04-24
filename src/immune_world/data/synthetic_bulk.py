"""Dirichlet synthetic bulk generation for cell-type deconvolution.

Ref: Sec. 4.6 — "we randomly select 500 single cells from a pool of available cell types ...
binarisation function set at a threshold of five per-cent for each cell type used in generating
the F1 score".

This module operates on plain tensors to keep the deconvolution pipeline runnable in
CI without `anndata` / `scanpy`. `SyntheticBulkDataset.from_anndata(...)` is provided in
`immune_world.data.anndata_io` and is only exercised when real single-cell data is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torch import Tensor


class SyntheticBulkDataset(Dataset[dict[str, "Tensor"]]):
    """Dirichlet-sampled pseudo-bulk mixtures.

    Args:
        single_cell_matrix: (N_cells, G) expression matrix — the reference cell pool.
        cell_type_ids: (N_cells,) integer cell-type assignment in [0, n_cell_types).
        n_cell_types: how many cell types to deconvolve.
        n_mixtures: total synthetic bulks to generate (e.g. 500 per Sec. 4.6).
        cells_per_mixture: cells pooled into each synthetic bulk (paper uses 500).
        dirichlet_alpha: concentration parameter for the Dirichlet prior over cell-type
            proportions (a uniform simplex prior at 1.0).
        seed: deterministic seed for reproducibility.
    """

    def __init__(
        self,
        single_cell_matrix: Tensor,
        cell_type_ids: Tensor,
        n_cell_types: int = 8,
        n_mixtures: int = 500,
        cells_per_mixture: int = 500,
        dirichlet_alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        if single_cell_matrix.dim() != 2:
            raise ValueError(
                f"expected (N_cells, G) single_cell_matrix, "
                f"got shape {tuple(single_cell_matrix.shape)}"
            )
        if cell_type_ids.shape[0] != single_cell_matrix.shape[0]:
            raise ValueError("cell_type_ids length must equal N_cells")

        self.matrix = single_cell_matrix
        self.cell_type_ids = cell_type_ids.long()
        self.n_cell_types = n_cell_types
        self.n_mixtures = n_mixtures
        self.cells_per_mixture = cells_per_mixture
        self.dirichlet_alpha = dirichlet_alpha

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        # Precompute per-cell-type row indices for fast sampling.
        self._by_type: list[Tensor] = [
            torch.nonzero(self.cell_type_ids == c, as_tuple=False).squeeze(-1)
            for c in range(n_cell_types)
        ]

    def __len__(self) -> int:
        return self.n_mixtures

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if not 0 <= idx < self.n_mixtures:
            raise IndexError(f"mixture index {idx} out of range [0, {self.n_mixtures})")
        props = self._sample_proportions()  # (n_cell_types,)
        per_type_counts = self._round_counts(props)
        pooled = self._pool_cells(per_type_counts)  # (G,)
        return {"bulk": pooled, "proportions": props}

    def _sample_proportions(self) -> Tensor:
        alphas = torch.full((self.n_cell_types,), self.dirichlet_alpha)
        # torch's Gamma supports a Generator only via torch.distributions; we hand-roll Dirichlet
        # via normalised Gammas to keep the seed deterministic.
        gamma = torch.empty(self.n_cell_types)
        for c in range(self.n_cell_types):
            gamma[c] = torch._standard_gamma(alphas[c : c + 1], generator=self._rng)
        return gamma / gamma.sum()

    def _round_counts(self, proportions: Tensor) -> list[int]:
        counts = (proportions * self.cells_per_mixture).round().long()
        # Fix rounding drift so counts sum to cells_per_mixture exactly.
        drift = int(self.cells_per_mixture - int(counts.sum()))
        if drift != 0:
            top = int(torch.argmax(counts))
            counts[top] += drift
        return [int(c) for c in counts]

    def _pool_cells(self, per_type_counts: list[int]) -> Tensor:
        selected: list[Tensor] = []
        for c, n_sample in enumerate(per_type_counts):
            if n_sample == 0 or self._by_type[c].numel() == 0:
                continue
            idx = torch.randint(0, int(self._by_type[c].numel()), (n_sample,), generator=self._rng)
            selected.append(self.matrix[self._by_type[c][idx]])
        if not selected:
            return torch.zeros(self.matrix.shape[1], dtype=self.matrix.dtype)
        return torch.cat(selected, dim=0).mean(dim=0)
