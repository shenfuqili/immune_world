"""CLI: fetch + cache datasets; write SHA-256 manifests."""

from __future__ import annotations

import argparse

from immune_world import get_logger

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="immune-world-prep-data",
        description="Fetch and preprocess paper datasets; write SHA-256 manifests.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        help="dataset names: norman, adamson, replogle_k562, replogle_rpe1, pancreas, "
        "dentate_gyrus, bone_marrow, icbatlas, pretrain_corpus, or 'all'",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        _LOG.info("dry-run: datasets=%s output=%s", args.datasets, args.output_dir)
        return 0
    raise NotImplementedError("data-prep CLI wiring")
