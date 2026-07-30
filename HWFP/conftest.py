"""Pytest configuration for HWFP/.

No `sys.path` mutation here (REQ-14, architecture-boundaries): the repo root
is already made importable via `pyproject.toml`'s `[tool.pytest.ini_options]
pythonpath` setting, which pytest applies natively before collection.
"""

from __future__ import annotations
