"""Immunotherapy-response metric — AUC under leave-one-cohort-out cross-validation + 95% CI.

Ref: Sec. 2.5 — "14 independent cohorts ... AUC of 0.891 ± 0.014 using leave-one-cohort-out
cross-validation".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AUCResult(NamedTuple):
    auc: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def _roc_auc(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    """Mann-Whitney U implementation of ROC-AUC; matches `sklearn.metrics.roc_auc_score`."""
    if np.unique(labels).size < 2:
        return float("nan")
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    total = 0.0
    # Ranks with ties get 0.5 credit each side.
    for s in pos:
        total += float(np.sum(s > neg) + 0.5 * np.sum(s == neg))
    return float(total / (pos.size * neg.size))


def auc_loco(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    cohorts: NDArray[np.int64],
    n_bootstrap: int = 1000,
) -> AUCResult:
    """Leave-one-cohort-out AUC aggregated across folds with bootstrap-1000 95% CI.

    Point estimate is the mean per-cohort AUC; CI comes from percentile bootstrap at the cohort
    level.
    """
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    labels_ = np.asarray(labels, dtype=np.int64).ravel()
    cohorts_ = np.asarray(cohorts, dtype=np.int64).ravel()
    if not (scores_.shape == labels_.shape == cohorts_.shape):
        raise ValueError("scores / labels / cohorts must share shape")

    unique_cohorts = np.unique(cohorts_)
    per_cohort_auc = np.array(
        [_roc_auc(scores_[cohorts_ == c], labels_[cohorts_ == c]) for c in unique_cohorts]
    )
    valid = ~np.isnan(per_cohort_auc)
    per_cohort_auc = per_cohort_auc[valid]
    if per_cohort_auc.size == 0:
        return AUCResult(auc=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n_bootstrap=0)

    rng = np.random.default_rng(42)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        resample = rng.choice(per_cohort_auc, size=per_cohort_auc.size, replace=True)
        boot[b] = resample.mean()

    return AUCResult(
        auc=float(per_cohort_auc.mean()),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        n_bootstrap=n_bootstrap,
    )
