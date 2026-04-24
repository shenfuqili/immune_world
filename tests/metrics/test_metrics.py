"""Unit tests for metrics (Tables 1 / 2 / S1 / S8)."""

from __future__ import annotations

import numpy as np
import torch

from immune_world.metrics.deconvolution import f1_at_threshold
from immune_world.metrics.icb import _roc_auc, auc_loco
from immune_world.metrics.perturbation import (
    mean_absolute_error,
    pearson_r,
    precision_recall_specificity,
)
from immune_world.metrics.statistics import bootstrap_ci, delong_test, paired_t_test
from immune_world.metrics.trajectory import cbdir, dtw_distance

# -- Pearson r, MAE, PRS ---------------------------------------------------


def test_pearson_r_on_identical_vectors_equals_one() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    r = pearson_r(x, x)
    assert abs(float(r) - 1.0) < 1e-5


def test_pearson_r_on_negated_equals_minus_one() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    r = pearson_r(x, -x)
    assert abs(float(r) - (-1.0)) < 1e-5


def test_mae_zero_on_equal_inputs() -> None:
    x = torch.randn(4, 5)
    assert float(mean_absolute_error(x, x)) == 0.0


def test_precision_recall_specificity_simple() -> None:
    pred = torch.tensor([0.9, 0.1, 0.8, 0.2])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    m = precision_recall_specificity(pred, target, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0


# -- Trajectory metrics ----------------------------------------------------


def test_cbdir_on_monotone_ordering_is_positive_one() -> None:
    pred = np.arange(10, dtype=np.float64)
    bounds = np.repeat(np.arange(5), 2).astype(np.int64)  # 0,0,1,1,2,2,3,3,4,4
    assert cbdir(pred, bounds) == 1.0


def test_cbdir_on_reversed_ordering_is_minus_one() -> None:
    pred = np.arange(10, 0, -1, dtype=np.float64)
    bounds = np.repeat(np.arange(5), 2).astype(np.int64)
    assert cbdir(pred, bounds) == -1.0


def test_dtw_on_identical_series_is_zero() -> None:
    x = np.sin(np.linspace(0, 2 * np.pi, 20))
    assert dtw_distance(x, x) == 0.0


def test_dtw_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(10)
    b = rng.standard_normal(12)
    assert abs(dtw_distance(a, b) - dtw_distance(b, a)) < 1e-8


# -- Deconvolution ---------------------------------------------------------


def test_f1_at_threshold_on_perfect_matches_equals_one() -> None:
    pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    true = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert abs(float(f1_at_threshold(pred, true, threshold=0.5)) - 1.0) < 1e-5


# -- AUC (LOCO + CI) -------------------------------------------------------


def test_roc_auc_on_perfectly_separated_scores_is_one() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assert _roc_auc(scores, labels) == 1.0


def test_auc_loco_returns_mean_with_ci() -> None:
    rng = np.random.default_rng(0)
    # Two cohorts, each with perfect separation → per-cohort AUC = 1.
    scores = np.concatenate(
        [
            rng.uniform(0, 0.4, 10),
            rng.uniform(0.6, 1.0, 10),
            rng.uniform(0, 0.4, 10),
            rng.uniform(0.6, 1.0, 10),
        ]
    )
    labels = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10, dtype=np.int64)
    cohorts = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    res = auc_loco(scores, labels, cohorts, n_bootstrap=200)
    assert abs(res.auc - 1.0) < 1e-9
    assert res.ci_low <= res.auc <= res.ci_high
    assert res.n_bootstrap == 200


# -- Statistical tests -----------------------------------------------------


def test_delong_test_identical_scores_has_p_equals_one() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    scores = rng.standard_normal(6)
    res = delong_test(labels, scores, scores)
    assert res.p_value == 1.0


def test_paired_t_test_matches_scipy_on_identical_arrays() -> None:
    a = np.array([0.1, 0.2, 0.3, 0.4])
    res = paired_t_test(a, a)
    assert np.isnan(res.statistic) or res.p_value == 1.0


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)

    def metric(arr: np.ndarray) -> float:
        return float(arr.mean())

    point, lo, hi = bootstrap_ci(metric, x, n_resamples=200, ci=0.95, seed=42)
    assert lo <= point <= hi
