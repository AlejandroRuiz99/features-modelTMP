"""
overlay.fill_actuals — CLI to backfill actual_fouls in overlay log files.

Usage:
    python -m overlay.fill_actuals <log_path> <fouls> [--force]

Arguments:
    log_path    Path to the overlay JSON log file.
    fouls       Actual foul count (integer).

Options:
    --force     Allow overwriting an existing non-null actual_fouls value.

Behaviour:
    - Reads the JSON log at log_path.
    - Sets actual_fouls to the provided integer.
    - Idempotent: if actual_fouls already equals the supplied value, exits 0.
    - Refuses to overwrite a non-null existing value unless --force is given.
    - Rejects non-numeric fouls arguments (exits 1).
    - Rejects missing log files (exits 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill actual_fouls in an overlay JSON log file."
    )
    parser.add_argument("log_path", help="Path to the overlay JSON log file.")
    parser.add_argument("fouls", help="Actual foul count (integer).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing non-null actual_fouls value.",
    )
    return parser.parse_args(argv)


def fill_actuals(log_path: Path, fouls: int, *, force: bool = False) -> None:
    """Update actual_fouls in the JSON log file.

    Args:
        log_path: Path to the overlay JSON log file.
        fouls:    Actual foul count to store.
        force:    If True, allow overwriting an existing non-null value.

    Raises:
        FileNotFoundError: If log_path does not exist.
        ValueError:        If refusing to overwrite without force.
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    record = json.loads(log_path.read_text(encoding="utf-8"))

    existing = record.get("actual_fouls")

    # Idempotent: same value → no-op
    if existing == fouls:
        return

    # Guard: non-null existing value requires --force
    if existing is not None and not force:
        raise ValueError(
            f"actual_fouls is already set to {existing!r}. Use --force to overwrite."
        )

    record["actual_fouls"] = fouls
    log_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Validate fouls argument
    try:
        fouls_int = int(args.fouls)
    except ValueError:
        print(
            f"ERROR: fouls must be an integer, got {args.fouls!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    log_path = Path(args.log_path)

    try:
        fill_actuals(log_path, fouls_int, force=args.force)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
