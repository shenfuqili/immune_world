"""Pretraining orchestrator for the 4× A100 / 288 GPU-h run.

Ref: Sec. 4.6 — 12.4 M cells, 200 epochs, batch 512, AdamW lr 1e-4 + 5k warmup + cosine to 1e-6.

The per-batch step uses the composite objective (Eq. 9). Real pretraining corpus assembly lives
in `data/pretrain_corpus.py` and is still a stub — the training loop here is fully runnable as
soon as that loader yields `dict[str, Tensor]` batches in the expected schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from immune_world.losses.composite import CompositeObjective
from immune_world.models.world_simulator import ImmuneWorld
from immune_world.training.optim import build_optimizer, build_scheduler
from immune_world.training.trainer import Trainer, TrainerConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor


@dataclass
class PretrainConfig:
    n_genes: int = 2000
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    lr: float = 1.0e-4
    warmup_steps: int = 5000
    total_steps: int = 100_000
    lr_min: float = 1.0e-6
    lambda_recon: float = 0.5
    lambda_pert: float = 0.3
    grad_clip: float | None = 1.0


class Pretrainer:
    """Multi-task pretraining using the composite objective (Eq. 9)."""

    def __init__(self, cfg: PretrainConfig) -> None:
        self.cfg = cfg
        self.model = ImmuneWorld(
            n_genes=cfg.n_genes,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
        )
        self.loss = CompositeObjective(lambda_recon=cfg.lambda_recon, lambda_pert=cfg.lambda_pert)

    def step(self, model: ImmuneWorld, batch: dict[str, Tensor]) -> Tensor:
        x = batch["x"]
        pt = batch["pseudo_time"]
        target_next = batch["target_next"]
        recon_target = batch["recon_target"]
        recon_mask = batch["recon_mask"]
        pert_target = batch["pert_target"]
        pert_mask = batch["pert_mask"]

        out = model(x, pt, perturbation=None)
        composite = self.loss(
            pred_next=out["trajectory_pred"],
            target_next=target_next,
            recon_pred=out["perturbation_pred"],
            recon_target=recon_target,
            recon_mask=recon_mask,
            pert_pred=out["perturbation_pred"],
            pert_target=pert_target,
            pert_mask=pert_mask,
        )
        total: Tensor = composite["total"]
        return total

    def fit(self, batches: Iterable[dict[str, Tensor]]) -> list[float]:
        optim = build_optimizer(self.model.parameters(), lr=self.cfg.lr)
        sched = build_scheduler(
            optim,
            schedule="cosine",
            warmup_steps=self.cfg.warmup_steps,
            total_steps=self.cfg.total_steps,
            lr_min=self.cfg.lr_min,
        )
        trainer = Trainer(
            self.model,
            optim,
            sched,
            step_fn=self.step,  # type: ignore[arg-type]
            cfg=TrainerConfig(max_steps=self.cfg.total_steps, grad_clip=self.cfg.grad_clip),
        )
        return trainer.fit(batches)
