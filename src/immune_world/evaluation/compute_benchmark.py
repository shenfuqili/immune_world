"""Compute-cost benchmark — Table 5 columns (params / GPU-h / throughput / memory)."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from torch.nn import Module


class ComputeReport(TypedDict):
    params_m: float
    pretrain_gpu_hours: float
    inference_cells_per_sec: float
    peak_memory_gb: float
    multi_task: bool


def benchmark_compute(
    model: Module, *, batch_size: int = 256, device: str = "cuda"
) -> ComputeReport:
    raise NotImplementedError("Table 5")
