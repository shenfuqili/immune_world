"""Cell-type deconvolution evaluation — Table 1 F1 + Table 2 per-cell-type F1."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig


def evaluate_deconvolution(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 1 deconv column + Table 2")


def evaluate_deconv_by_celltype(cfg: DictConfig) -> dict[str, float]:
    raise NotImplementedError("Table 2 per-cell-type F1")
