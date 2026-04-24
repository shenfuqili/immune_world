"""Statistical-significance tests used in Table S8.

Ref: Table S8 — "DeLong test for two correlated ROC curves ... Pearson r and F1 comparisons, we
use paired t-tests across 3 independent runs ... for trajectory metrics (CBDir, DTW), we use
bootstrap resampling (1,000 iterations)".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray  # runtime-used type alias below
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Callable


class PValueResult(NamedTuple):
    statistic: float
    p_value: float
    ci_low: float | None
    ci_high: float | None


def _midrank(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Midrank transform used by DeLong's fast variance estimator."""
    order = np.argsort(x, kind="mergesort")
    ranked = np.empty_like(x, dtype=np.float64)
    i = 0
    n = x.size
    while i < n:
        j = i
        while j < n - 1 and x[order[j]] == x[order[j + 1]]:
            j += 1
        mid = 0.5 * (i + j) + 1.0
        ranked[order[i : j + 1]] = mid
        i = j + 1
    return ranked


def _delong_components(
    labels: NDArray[np.int64], scores_a: NDArray[np.float64], scores_b: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Return (auc_a, auc_b, variance_diff) via DeLong's fast U-statistic approach."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    if not pos_mask.any() or not neg_mask.any():
        raise ValueError("DeLong requires both positive and negative labels")

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    Arr = NDArray[np.float64]

    def _components(scores: Arr) -> tuple[float, Arr, Arr]:
        pos = scores[pos_mask]
        neg = scores[neg_mask]
        tx = _midrank(neg)
        ty = _midrank(pos)
        tz = _midrank(np.concatenate([pos, neg]))
        v10 = (tz[:n_pos] - ty) / n_neg
        v01 = 1.0 - (tz[n_pos:] - tx) / n_pos
        auc = float(v10.mean())
        return auc, v10, v01

    auc_a, v10_a, v01_a = _components(scores_a)
    auc_b, v10_b, v01_b = _components(scores_b)

    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    cov = s10 / n_pos + s01 / n_neg
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    return auc_a, auc_b, var_diff


def delong_test(
    labels: NDArray[np.int64],
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
) -> PValueResult:
    """DeLong test for two correlated ROC curves — two-sided p-value."""
    labels_ = np.asarray(labels, dtype=np.int64).ravel()
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    if not (labels_.shape == a.shape == b.shape):
        raise ValueError("labels / scores_a / scores_b must share shape")

    auc_a, auc_b, var_diff = _delong_components(labels_, a, b)
    if var_diff <= 0.0:
        return PValueResult(statistic=0.0, p_value=1.0, ci_low=None, ci_high=None)
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return PValueResult(statistic=float(z), p_value=p, ci_low=None, ci_high=None)


def paired_t_test(a: NDArray[np.float64], b: NDArray[np.float64]) -> PValueResult:
    """Two-sided paired t-test; used for Pearson r / F1 across seeds {42, 123, 456}."""
    a_ = np.asarray(a, dtype=np.float64).ravel()
    b_ = np.asarray(b, dtype=np.float64).ravel()
    if a_.shape != b_.shape:
        raise ValueError(f"shape mismatch: a {a_.shape} vs b {b_.shape}")
    result = stats.ttest_rel(a_, b_, alternative="two-sided")
    return PValueResult(
        statistic=float(result.statistic), p_value=float(result.pvalue), ci_low=None, ci_high=None
    )


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *args: NDArray[np.float64],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) under bootstrap resampling of the first argument."""
    if not args:
        raise ValueError("bootstrap_ci requires at least one positional array argument")
    primary = np.asarray(args[0])
    rest = tuple(np.asarray(a) for a in args[1:])
    point = float(metric_fn(primary, *rest))

    rng = np.random.default_rng(seed)
    n = primary.shape[0]
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_primary = primary[idx]
        resampled_rest = tuple(a[idx] for a in rest)
        draws[i] = float(metric_fn(resampled_primary, *resampled_rest))

    alpha = (1.0 - ci) / 2.0 * 100.0
    return point, float(np.percentile(draws, alpha)), float(np.percentile(draws, 100.0 - alpha))
