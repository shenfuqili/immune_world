"""CLI producing Table 5 compute-cost comparison."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce Table 5.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.parse_args(argv)
    raise NotImplementedError("Table 5")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
