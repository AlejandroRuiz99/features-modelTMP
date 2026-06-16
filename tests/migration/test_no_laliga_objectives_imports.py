"""
tests/migration/test_no_laliga_objectives_imports.py — D17 migration gate.

Asserts zero references to fetch_laliga_objectives remain in production code.
This test MUST pass after the D17 migration is complete.
"""

from __future__ import annotations

import re
from pathlib import Path

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories to skip (test files themselves, __pycache__, .git)
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
}

# Pattern to search for
PATTERN = re.compile(r"fetch_laliga_objectives")


def test_no_fetch_laliga_objectives_in_production_code() -> None:
    """No .py file in the project should reference fetch_laliga_objectives.

    This test file itself is excluded from the search.
    """
    violations: list[str] = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip __pycache__ directories
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue

        # Skip this test file
        if py_file.name == "test_no_laliga_objectives_imports.py":
            continue

        content = py_file.read_text(encoding="utf-8", errors="ignore")
        if PATTERN.search(content):
            violations.append(str(py_file.relative_to(PROJECT_ROOT)))

    assert violations == [], (
        f"Found fetch_laliga_objectives references in: {violations}. "
        f"This function was removed in D17 migration — objectives now come from "
        f"narrative YAML via overlay.objective.inject_objectives_into_state()."
    )
