"""Trajectory-reconstruction metrics — CBDir (higher better) + DTW (lower better).

Ref: Sec. 2.3 — "Cross-Boundary Direction correctness (CBDir)" + "Dynamic Time Warping (DTW)".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cbdir(
    predicted_ordering: NDArray[np.float64],
    true_boundaries: NDArray[np.int64],
) -> float:
    """Cross-boundary direction correctness.

    Given a 1-D predicted pseudo-time ordering (one scalar per cell, monotone-increasing is
    "forward") and a 1-D array `true_boundaries` that assigns each cell to an ordered cluster id
    (0, 1, 2 ...), CBDir = (n_forward_transitions − n_reverse_transitions) / n_cross_boundary.

    Returns a value in [−1, 1]; identical ordering → +1, reversed ordering → −1. Cells in the same
    boundary cluster do not contribute (not a boundary crossing).
    """
    pred = np.asarray(predicted_ordering, dtype=np.float64).ravel()
    bounds = np.asarray(true_boundaries, dtype=np.int64).ravel()
    if pred.shape != bounds.shape:
        raise ValueError(
            f"shape mismatch: predicted_ordering {pred.shape} vs true_boundaries {bounds.shape}"
        )
    if pred.size < 2:
        raise ValueError("need at least 2 cells to compute CBDir")

    order = np.argsort(pred)
    bounds_sorted = bounds[order]
    diffs = np.diff(bounds_sorted)
    forward = int(np.sum(diffs > 0))
    reverse = int(np.sum(diffs < 0))
    total = forward + reverse
    return 0.0 if total == 0 else (forward - reverse) / total


def dtw_distance(predicted: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """Pure-numpy DTW distance with squared-Euclidean local cost (lower is better).

    `predicted` and `reference` can be 1-D (scalar series) or 2-D `(T, D)` multivariate series.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[:, None]
    if ref.ndim == 1:
        ref = ref[:, None]
    if pred.shape[1] != ref.shape[1]:
        raise ValueError(
            f"feature dim mismatch: predicted {pred.shape[1]} vs reference {ref.shape[1]}"
        )

    n, m = pred.shape[0], ref.shape[0]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.sum((pred[i - 1] - ref[j - 1]) ** 2))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[n, m] / (n + m)))
