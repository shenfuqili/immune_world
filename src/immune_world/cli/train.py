"""`python -m immune_world.cli.train experiment=<name>` → Pretrainer or FineTuner dispatch."""

from __future__ import annotations

import argparse

from immune_world import get_logger
from immune_world.utils.config import load_config
from immune_world.utils.paths import resolve_dataset_path

_LOG = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Hydra-compatible thin argparse wrapper used for `--help` and smoke tests."""
    parser = argparse.ArgumentParser(prog="immune-world-train", description="Train ImmuneWorld.")
    parser.add_argument("experiment", nargs="?", default="main", help="experiment config name")
    parser.add_argument("--config-dir", default="configs", help="root of configs tree")
    parser.add_argument(
        "--phase",
        choices=["pretrain", "finetune"],
        default="pretrain",
        help="which phase of training to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="parse config then exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point registered in `pyproject.toml [project.scripts]`."""
    args = build_parser().parse_args(argv)
    cfg_override = f"{args.config_dir}/experiment/{args.experiment}.yaml"
    cfg_path = resolve_dataset_path(f"experiment/{args.experiment}.yaml", override=cfg_override)
    if args.dry_run:
        _LOG.info("dry-run: experiment=%s phase=%s cfg=%s", args.experiment, args.phase, cfg_path)
        return 0

    if not cfg_path.is_file():
        _LOG.error("config not found: %s", cfg_path)
        return 2
    cfg = load_config(cfg_path)
    _LOG.info(
        "loaded config: experiment=%s phase=%s keys=%s",
        args.experiment,
        args.phase,
        list(cfg),
    )
    _LOG.error(
        "full training run requires data/pretrain_corpus.py to be implemented; "
        "run `make smoke` or "
        "`pytest -q tests/training/test_training_smoke.py` for the two-step smoke."
    )
    return 1
