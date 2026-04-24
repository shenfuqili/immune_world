"""Perturbation-prediction metrics — Pearson r, MAE, precision/recall/specificity.

Ref: Table 1 (Pearson r), Table S1 (precision / recall / specificity / MAE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def pearson_r(pred: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    """Pearson correlation along `dim`. Returns a scalar if inputs are 1-D."""
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    pred_c = pred - pred.mean(dim=dim, keepdim=True)
    target_c = target - target.mean(dim=dim, keepdim=True)
    num = (pred_c * target_c).sum(dim=dim)
    denom = torch.sqrt(pred_c.pow(2).sum(dim=dim) * target_c.pow(2).sum(dim=dim)).clamp(min=1e-12)
    return num / denom


def mean_absolute_error(pred: Tensor, target: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    return (pred - target).abs().mean()


def precision_recall_specificity(
    pred: Tensor, target: Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Binary precision / recall / specificity after thresholding `pred` and `target`."""
    pred_bin = (pred >= threshold).to(torch.bool)
    target_bin = (target >= threshold).to(torch.bool)

    tp = float((pred_bin & target_bin).sum().item())
    fp = float((pred_bin & ~target_bin).sum().item())
    fn = float((~pred_bin & target_bin).sum().item())
    tn = float((~pred_bin & ~target_bin).sum().item())

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    return {
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
    }
