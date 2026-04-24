"""CLI producing Table S6 — corpus-size × epoch grid (3.1M/6.2M/9.3M/12.4M × {10..200} epochs)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce Table S6.")
    parser.add_argument("--output-csv", required=True)
    parser.parse_args(argv)
    raise NotImplementedError("Table S6")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
