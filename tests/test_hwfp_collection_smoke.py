"""Smoke test: HWFP tests are discoverable via pytest collection."""

from __future__ import annotations

import subprocess
import sys


def test_hwfp_tests_discovered() -> None:
    """Verify that pytest collects >= 100 items from the HWFP test suite."""
    # --override-ini=addopts= clears the repo's "-v --tb=short" so that -q
    # produces flat "path::test_name" lines instead of tree format.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--override-ini=addopts=",
            "HWFP",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"pytest --collect-only HWFP exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    collected = [line for line in output.splitlines() if "::" in line]
    assert len(collected) >= 100, (
        f"Expected >=100 collected items, got {len(collected)}.\nOutput:\n{output}"
    )
    assert any("HWFP/tests/unit/" in line for line in output.splitlines()), (
        f"No HWFP/tests/unit/ path in collected output.\nOutput:\n{output}"
    )
