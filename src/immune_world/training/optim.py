"""Optimiser + learning-rate-schedule builders.

Ref: Sec. 2.1 — AdamW, lr 3e-4 (fine-tune), cosine annealing, weight-decay 0.01.
Ref: Sec. 4.6 — AdamW, lr 1e-4 (pretrain), 5000-step linear warmup, cosine decay to 1e-6.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.nn import Parameter
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer


def build_optimizer(
    params: Iterable[Parameter],
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
) -> Optimizer:
    """Construct the AdamW optimiser used by every training run (pretrain + fine-tune)."""
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: Optimizer,
    *,
    schedule: str = "cosine",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    lr_min: float = 0.0,
) -> LRScheduler:
    """Linear warmup → cosine decay to `lr_min`.

    Supported `schedule` values: ``"cosine"`` (paper default) or ``"constant"`` (smoke tests).
    `total_steps` is the number of optimiser steps *including* warmup; required for cosine.
    """
    if schedule not in {"cosine", "cosine_annealing", "constant"}:
        raise ValueError(f"unsupported schedule={schedule!r}")

    base_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 1.0
    if base_lr <= 0.0:
        raise ValueError("optimizer base learning rate must be positive")
    decay_floor = lr_min / base_lr if base_lr > 0 else 0.0

    def _lr_lambda(step: int) -> float:
        if schedule == "constant":
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        if total_steps is None:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(decay_floor + (1.0 - decay_floor) * cosine)

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)
