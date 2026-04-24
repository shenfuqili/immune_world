"""`python -m immune_world.cli.export_onnx checkpoint=<path>` — ONNX export of the main model."""

from __future__ import annotations

import argparse

from immune_world import get_logger

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immune-world-export", description="Export ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        _LOG.info("dry-run: ckpt=%s opset=%d", args.checkpoint, args.opset)
        return 0
    raise NotImplementedError("ONNX export wiring")
