"""CLI producing Table S4 — worst-10 Norman/K562/ICB samples with error categorisation."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce Table S4.")
    parser.add_argument("--results", required=True, help="path to evaluation JSON")
    parser.add_argument("--output-csv", required=True)
    parser.parse_args(argv)
    raise NotImplementedError("Table S4")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
