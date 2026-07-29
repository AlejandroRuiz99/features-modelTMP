"""T8.5 — CLI smoke test: predict.py exits 0 and emits match_id in JSON output."""

from __future__ import annotations

import subprocess
import sys


def test_cli_predict_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "HWFP.cli.predict", "--match-id", "M1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "M1" in result.stdout
