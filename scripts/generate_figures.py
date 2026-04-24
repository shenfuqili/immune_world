"""CLI dispatching to per-figure reproduction routines (Fig. 1, 5, 6, 7, 8)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce paper figures 1/5/6/7/8.")
    parser.add_argument("--figure", choices=["1", "5", "6", "7", "8", "all"], default="all")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", default="assets")
    parser.parse_args(argv)
    raise NotImplementedError("figure reproduction")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
