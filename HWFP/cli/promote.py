"""CLI entry point: promote a candidate model to production."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HWFP model promotion CLI")
    parser.add_argument("--model-id", required=True, help="Model identifier to promote")
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
