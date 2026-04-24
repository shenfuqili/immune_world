"""`python -m immune_world.cli.eval experiment=<name>` → runs evaluation, writes metrics JSON."""

from __future__ import annotations

import argparse

from immune_world import get_logger

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immune-world-eval", description="Evaluate ImmuneWorld.")
    parser.add_argument("experiment", nargs="?", default="main")
    parser.add_argument("--checkpoint", required=False, default=None)
    parser.add_argument("--output-json", required=False, default="metrics.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        _LOG.info("dry-run: experiment=%s", args.experiment)
        return 0
    raise NotImplementedError("evaluation CLI wiring")
