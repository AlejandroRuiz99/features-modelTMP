"""
audit.audit_root_files -- Root-level stray file detector.

Walks the repo root (one level only -- not recursive) and flags files
matching forbidden patterns. These patterns correspond to artifacts that
should live inside runs/{id}/ folders, not at the repo root.

Usage:
    python -m audit.audit_root_files [repo_root]

Exit code:
    0 -- no violations
    1 -- violations found (prints them to stdout)
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

__all__ = ["FORBIDDEN_PATTERNS", "audit_root_files"]

# Forbidden root-level file patterns (from .gitignore additions in Batch 4)
# These match only files directly in repo root (not in subdirs).
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "partidos_*.json",
    "resultados_*.json",
    "features_dump*.json",
    "informe_*.pdf",
    "nul",
)


def audit_root_files(
    repo_root: Path,
    gitignore_path: Path | None = None,
) -> list[str]:
    """Check repo root for stray prediction artifacts.

    Only checks files directly in repo_root (non-recursive).

    Args:
        repo_root: Repository root directory to check.
        gitignore_path: Optional path to .gitignore (reserved for Batch 4 enforcement).
            In Batch 1, patterns are hardcoded in FORBIDDEN_PATTERNS.

    Returns:
        List of relative path strings for files matching forbidden patterns.
        Empty list means the repo root is clean.
    """
    if not repo_root.exists():
        return []

    violations: list[str] = []

    for entry in repo_root.iterdir():
        # Only check files directly at root level (not subdirs)
        if not entry.is_file():
            continue

        name = entry.name
        for pattern in FORBIDDEN_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                violations.append(str(entry.relative_to(repo_root)))
                break

    return violations


if __name__ == "__main__":
    # CLI: python -m audit.audit_root_files [repo_root]
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    found = audit_root_files(root)
    if found:
        print("AUDIT FAIL: Forbidden root-level files found:")
        for v in found:
            print(f"  - {v}")
        sys.exit(1)
    else:
        print("AUDIT PASS: No forbidden root-level files found.")
        sys.exit(0)
