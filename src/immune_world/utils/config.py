"""Hydra / OmegaConf loading and validation shim.

Keep a thin wrapper so modules can import `load_config` without depending on Hydra at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML file into an OmegaConf DictConfig; fails loudly if file is missing."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    loaded = OmegaConf.load(cfg_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"expected mapping at top level of {cfg_path}, got {type(loaded).__name__}")
    return loaded


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a DictConfig into a plain dict (for JSON dumping, etc.)."""
    return cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
