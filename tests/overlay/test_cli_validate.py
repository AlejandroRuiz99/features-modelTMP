"""
tests/overlay/test_cli_validate.py — Tests for --validate-narrative CLI flag.

Tests (3 cases — Strict TDD, all RED before flag is added to run_prediction.py):
  1. --validate-narrative valid.yaml exits 0 with "OK" in output
  2. --validate-narrative bad.yaml exits non-zero with parse error in output
  3. --validate-narrative missing.yaml exits non-zero with error

NOTE: These tests use subprocess.run to avoid polluting the current process.
      The flag must NOT invoke the prediction pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "narratives"
RUN_PREDICTION = Path(__file__).parent.parent.parent / "run_prediction.py"
PYTHON = sys.executable


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run run_prediction.py with the given args."""
    return subprocess.run(
        [PYTHON, str(RUN_PREDICTION), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestValidateNarrativeCLI:
    def test_valid_yaml_exits_zero(self) -> None:
        """--validate-narrative valid_full.yaml exits 0 and prints OK."""
        result = _run(["--validate-narrative", str(FIXTURES_DIR / "valid_full.yaml")])
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "OK" in combined or "valid" in combined.lower(), (
            f"Expected 'OK' or 'valid' in output, got:\n{combined}"
        )

    def test_bad_yaml_exits_nonzero(self) -> None:
        """--validate-narrative bad_yaml.yaml exits non-zero with error info."""
        result = _run(["--validate-narrative", str(FIXTURES_DIR / "bad_yaml.yaml")])
        assert result.returncode != 0, (
            f"Expected non-zero exit, got 0\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert (
            "bad_yaml" in combined or "error" in combined.lower() or "Error" in combined
        ), f"Expected error info in output, got:\n{combined}"

    def test_missing_yaml_exits_nonzero(self) -> None:
        """--validate-narrative nonexistent.yaml exits non-zero."""
        result = _run(["--validate-narrative", "this_file_does_not_exist.yaml"])
        assert result.returncode != 0, (
            f"Expected non-zero exit, got 0\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
