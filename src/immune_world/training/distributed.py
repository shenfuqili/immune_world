"""DDP init / barrier / cleanup helpers for the 4× A100 pretraining job.

Ref: Sec. 4.6 — "four A100 80 GB GPUs".
Single-GPU and CPU smoke paths short-circuit (return rank 0, world size 1).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank).

    When `RANK` / `WORLD_SIZE` environment variables are absent (single-process run) this returns
    (0, 1, 0) and does NOT initialise a process group.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup() -> None:
    """Tear down the process group; safe to call if never initialised."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
