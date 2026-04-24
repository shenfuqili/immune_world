"""Unified Trainer orchestrating optimizer / scheduler / AMP / DDP / checkpointing.

This is a thin, loop-aware orchestrator; task-specific loss / metric wiring happens inside
`pretrain.py` and `finetune.py`. Exists so the CLI has a single construction site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from immune_world import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.nn import Module
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

_LOG = get_logger(__name__)


@dataclass
class TrainerConfig:
    max_steps: int
    log_every: int = 10
    grad_clip: float | None = 1.0


class Trainer:
    """Generic inner loop; callers supply a `step_fn(batch) -> loss` closure."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        step_fn: Callable[[Module, object], torch.Tensor],
        cfg: TrainerConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.cfg = cfg

    def fit(self, batches: Iterable[object]) -> list[float]:
        """Run `max_steps` optimiser iterations and return the per-step loss trace."""
        self.model.train()
        losses: list[float] = []
        for step, batch in enumerate(batches):
            if step >= self.cfg.max_steps:
                break
            self.optimizer.zero_grad()
            loss = self.step_fn(self.model, batch)
            loss.backward()  # type: ignore[no-untyped-call]
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            losses.append(float(loss.detach()))
            if step % self.cfg.log_every == 0:
                _LOG.info("step %d loss=%.4f", step, losses[-1])
        return losses
