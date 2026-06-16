"""
tests/overlay/test_fill_actuals.py — TDD for overlay.fill_actuals (T6.5).

Tests:
  1. fill_actuals updates actual_fouls in an existing log file.
  2. Rejects non-numeric input → exits with non-zero code.
  3. Refuses to overwrite existing non-null actual without --force.
  4. --force flag allows overwriting an existing non-null actual.
  5. Idempotent: setting same value twice is allowed (no error).
  6. Non-existent log file → exits with non-zero code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PYTHON = sys.executable  # same Python interpreter running pytest


def _write_log(path: Path, actual_fouls: object = None) -> None:
    """Write a minimal log file with the given actual_fouls value."""
    record = {
        "timestamp": "2026-04-27T10:00:00Z",
        "match": {"home": "Espanol", "away": "Levante", "date": "2026-04-27"},
        "actual_fouls": actual_fouls,
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _run_fill(
    log_path: Path, fouls_arg: str, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run: python -m overlay.fill_actuals <log> <fouls> [extra_args]"""
    cmd = [_PYTHON, "-m", "overlay.fill_actuals", str(log_path), fouls_arg]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFillActuals:
    def test_updates_actual_fouls_in_log_file(self, tmp_path: Path) -> None:
        """fill_actuals sets actual_fouls to the provided integer."""
        log_path = tmp_path / "match.json"
        _write_log(log_path)

        result = _run_fill(log_path, "27")

        assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
        content = json.loads(log_path.read_text(encoding="utf-8"))
        assert content["actual_fouls"] == 27

    def test_rejects_non_numeric_input(self, tmp_path: Path) -> None:
        """Non-numeric fouls argument → exits with non-zero code."""
        log_path = tmp_path / "match.json"
        _write_log(log_path)

        result = _run_fill(log_path, "abc")

        assert result.returncode != 0, "Expected non-zero exit for non-numeric input"

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        """Refuses to overwrite existing non-null actual_fouls without --force."""
        log_path = tmp_path / "match.json"
        _write_log(log_path, actual_fouls=25)

        result = _run_fill(log_path, "27")

        assert result.returncode != 0, (
            "Expected non-zero exit when overwriting existing non-null actual without --force"
        )
        # File should be unchanged
        content = json.loads(log_path.read_text(encoding="utf-8"))
        assert content["actual_fouls"] == 25

    def test_force_flag_allows_overwrite(self, tmp_path: Path) -> None:
        """--force flag allows overwriting an existing non-null actual_fouls."""
        log_path = tmp_path / "match.json"
        _write_log(log_path, actual_fouls=25)

        result = _run_fill(log_path, "27", ["--force"])

        assert result.returncode == 0, f"Non-zero exit with --force: {result.stderr}"
        content = json.loads(log_path.read_text(encoding="utf-8"))
        assert content["actual_fouls"] == 27

    def test_idempotent_same_value(self, tmp_path: Path) -> None:
        """Setting actual_fouls to the same value twice is allowed (no error)."""
        log_path = tmp_path / "match.json"
        _write_log(log_path, actual_fouls=27)

        result = _run_fill(log_path, "27")

        assert result.returncode == 0, (
            f"Expected success for idempotent same-value write: {result.stderr}"
        )
        content = json.loads(log_path.read_text(encoding="utf-8"))
        assert content["actual_fouls"] == 27

    def test_nonexistent_log_file_exits_nonzero(self, tmp_path: Path) -> None:
        """Non-existent log file path → exits with non-zero code."""
        missing = tmp_path / "does_not_exist.json"

        result = _run_fill(missing, "27")

        assert result.returncode != 0, "Expected non-zero exit for missing file"
