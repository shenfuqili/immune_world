"""Cross-cancer generalisation — Table 4 leave-one-cancer-out + cohort-size scan."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run_loco_cancer(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 4 top")


def cohort_size_scan(cfg: DictConfig) -> dict[int, float]:
    raise NotImplementedError("Table 4 bottom")
