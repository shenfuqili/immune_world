"""Mixed-precision (AMP) helpers.

Paper does not explicitly state fp16 vs bf16 (see `docs/deviations.md`); default bf16 on A100
matches FlashAttention-2's preferred dtype (Sec. 2.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler


Precision = Literal["fp32", "fp16", "bf16"]


def build_grad_scaler(precision: Precision = "bf16") -> GradScaler | None:
    """Return a GradScaler for fp16 (needed for stable gradients); None for bf16/fp32."""
    if precision == "fp16" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


def autocast_dtype(precision: Precision) -> torch.dtype | None:
    """Return the `torch.dtype` to pass to `torch.autocast`; `None` disables autocast."""
    match precision:
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case "fp32":
            return None
