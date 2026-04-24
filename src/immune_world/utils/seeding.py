"""Deterministic seeding for every RNG involved in training / evaluation.

R4: single `set_seed(seed)` utility; seed stored in every checkpoint.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_cuda: bool = False) -> None:
    """Seed Python `random`, NumPy, CPU and CUDA torch RNGs in one call.

    Ref: Sec. 2.1 — "all experiments were conducted using random seed 42 ... seeds {42, 123, 456}".
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_cuda:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
