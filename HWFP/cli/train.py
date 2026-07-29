"""CLI entry point: train a foul prediction model."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HWFP model training CLI")
    parser.add_argument(
        "--mode",
        choices=["fake", "production"],
        default="fake",
        help="Container mode (default: fake)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    print("not implemented")
    return 1


if __name__ == "__main__":
    sys.exit(main())
