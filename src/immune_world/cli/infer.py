"""CLI: trajectory prediction or counterfactual perturbation simulation."""

from __future__ import annotations

import argparse

from immune_world import get_logger

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immune-world-infer", description="Run inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="AnnData .h5ad input path")
    parser.add_argument("--perturbation", default=None, help="JSON file with perturbation context")
    parser.add_argument("--output", required=True, help="AnnData .h5ad output path")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        _LOG.info("dry-run: ckpt=%s input=%s", args.checkpoint, args.input)
        return 0
    raise NotImplementedError("inference CLI wiring")
