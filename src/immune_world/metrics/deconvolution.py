"""Cell-type deconvolution metric — F1 at 5 % detection threshold.

Ref: Sec. 4.6 Deconvolution — "threshold of five per-cent for each cell type used in generating
the F1 score".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def f1_at_threshold(predicted_props: Tensor, true_props: Tensor, threshold: float = 0.05) -> Tensor:
    """Per-cell-type binarised F1, macro-averaged across cell types.

    `predicted_props` and `true_props` are `(N_mixtures, N_cell_types)` proportion matrices.
    """
    if predicted_props.shape != true_props.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(predicted_props.shape)} "
            f"vs true {tuple(true_props.shape)}"
        )
    if predicted_props.dim() != 2:
        raise ValueError(f"expected 2-D proportions matrix, got {predicted_props.dim()}-D")

    pred_bin = predicted_props >= threshold
    true_bin = true_props >= threshold

    tp = (pred_bin & true_bin).sum(dim=0).to(torch.float32)
    fp = (pred_bin & ~true_bin).sum(dim=0).to(torch.float32)
    fn = (~pred_bin & true_bin).sum(dim=0).to(torch.float32)

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1_per_type = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    return f1_per_type.mean()
