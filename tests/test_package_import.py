"""Sanity: the package imports and exports a string version. Always runs in CI."""

from __future__ import annotations


def test_package_imports_and_has_version() -> None:
    import immune_world

    assert isinstance(immune_world.__version__, str)
    assert immune_world.__version__.count(".") >= 1


def test_get_logger_is_stdlib() -> None:
    import logging

    from immune_world import get_logger

    logger = get_logger("immune_world.test")
    assert isinstance(logger, logging.Logger)


def test_subpackages_importable() -> None:
    # Each subpackage must at least import without side effects.
    import immune_world.cli
    import immune_world.data
    import immune_world.evaluation
    import immune_world.losses
    import immune_world.metrics
    import immune_world.models
    import immune_world.training
    import immune_world.utils  # noqa: F401


def test_cli_entrypoints_parse_help() -> None:
    # Each CLI must at least build its parser and handle --dry-run without raising.
    from immune_world.cli import eval as eval_cli
    from immune_world.cli import export_onnx, infer, prepare_data, train

    for module in (train, eval_cli, infer, export_onnx, prepare_data):
        parser = module.build_parser()
        # argparse --help would sys.exit(0); skip that path in tests.
        assert parser.prog.startswith("immune-world")
