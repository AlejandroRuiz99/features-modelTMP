"""CLI entry point: python -m training_data"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generator import run

_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "predictionModels" / "data" / "training.parquet"
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Genera dataset de entrenamiento (Parquet) desde Supabase",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Ruta del Parquet de salida (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)
    output_path = Path(args.output) if args.output else _DEFAULT_OUTPUT
    return 0 if run(output_path) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
