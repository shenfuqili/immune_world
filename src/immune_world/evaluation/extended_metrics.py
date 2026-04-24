"""Extended metrics — Table S1 precision / recall / specificity / MAE + per-dataset CBDir/DTW."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def compute_extended_metrics(
    preds: Tensor, targets: Tensor, dataset_ids: Tensor
) -> dict[str, dict[str, float]]:
    """Return per-dataset breakdown with the columns of Table S1."""
    raise NotImplementedError("Table S1")
