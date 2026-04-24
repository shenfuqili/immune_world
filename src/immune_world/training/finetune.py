"""Per-task fine-tuning orchestrator.

Ref: Sec. 2.1 + Sec. 4.6 — AdamW lr 3e-4, cosine annealing, weight-decay 0.01, batch 256,
patience 10, seeds {42, 123, 456}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from immune_world.training.optim import build_optimizer, build_scheduler
from immune_world.training.trainer import Trainer, TrainerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import torch
    from torch.nn import Module


TaskName = Literal["perturbation", "trajectory", "deconvolution", "icb"]


@dataclass
class FineTuneConfig:
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    total_steps: int = 10_000
    warmup_steps: int = 0
    lr_min: float = 0.0
    early_stopping_patience: int = 10
    grad_clip: float | None = 1.0


class FineTuner:
    """Attaches the appropriate task head to a pretrained backbone and trains to convergence."""

    def __init__(
        self,
        model: Module,
        task: TaskName,
        cfg: FineTuneConfig,
        step_fn: Callable[[Module, object], torch.Tensor],
    ) -> None:
        self.model = model
        self.task = task
        self.cfg = cfg
        self.step_fn = step_fn

    def fit(self, batches: Iterable[object]) -> list[float]:
        optim = build_optimizer(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        sched = build_scheduler(
            optim,
            schedule="cosine_annealing",
            warmup_steps=self.cfg.warmup_steps,
            total_steps=self.cfg.total_steps,
            lr_min=self.cfg.lr_min,
        )
        trainer = Trainer(
            self.model,
            optim,
            sched,
            step_fn=self.step_fn,
            cfg=TrainerConfig(max_steps=self.cfg.total_steps, grad_clip=self.cfg.grad_clip),
        )
        return trainer.fit(batches)
