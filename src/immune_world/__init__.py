"""ImmuneWorld — trajectory-aware foundation model for tumour-immune-microenvironment simulation."""

from __future__ import annotations

import logging

__version__: str = "0.1.0"


def get_logger(name: str = "immune_world") -> logging.Logger:
    """Return a stdlib logger configured once per process (R4: no bare print in library code)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["__version__", "get_logger"]
