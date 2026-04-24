"""Immunotherapy-response evaluation — Table 1 AUC + Table 2 per-cancer AUC (LOOCV)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def evaluate_icb(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 1 ICB column")


def evaluate_icb_by_cancer(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 2 per-cancer AUC with 95% CI")
