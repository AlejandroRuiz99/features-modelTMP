"""
tests/audit/test_audit_root_files.py — TDD for audit.audit_root_files

Tests (T5.1):
  1. Clean repo (no forbidden files) → audit returns empty list.
  2. Forbidden file present → audit returns list with that file.
  3. Multiple forbidden files → all returned.
  4. Nested subdirs (not root level) → NOT flagged.
  5. The function signature: audit_root_files(repo_root, gitignore_path=None) works.
  6. __main__ block: exit code 0 when clean, non-zero when violations found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from audit.audit_root_files import audit_root_files


class TestCleanRepo:
    def test_clean_root_returns_empty(self, tmp_path: Path) -> None:
        """No forbidden files → empty list."""
        result = audit_root_files(tmp_path)
        assert result == []

    def test_unrelated_files_not_flagged(self, tmp_path: Path) -> None:
        """Legitimate files like README.md are not flagged."""
        (tmp_path / "README.md").write_text("readme", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("# config", encoding="utf-8")
        result = audit_root_files(tmp_path)
        assert result == []


class TestForbiddenFiles:
    def test_partidos_json_flagged(self, tmp_path: Path) -> None:
        """partidos_*.json at root is flagged."""
        (tmp_path / "partidos_2026-05-04.json").write_text("{}", encoding="utf-8")
        result = audit_root_files(tmp_path)
        assert len(result) == 1
        assert "partidos_2026-05-04.json" in result[0]

    def test_resultados_json_flagged(self, tmp_path: Path) -> None:
        """resultados_*.json at root is flagged."""
        (tmp_path / "resultados_hoy.json").write_text("{}", encoding="utf-8")
        result = audit_root_files(tmp_path)
        assert any("resultados_hoy.json" in v for v in result)

    def test_features_dump_flagged(self, tmp_path: Path) -> None:
        """features_dump*.json at root is flagged."""
        (tmp_path / "features_dump.json").write_text("{}", encoding="utf-8")
        result = audit_root_files(tmp_path)
        assert any("features_dump.json" in v for v in result)

    def test_informe_pdf_flagged(self, tmp_path: Path) -> None:
        """informe_*.pdf at root is flagged."""
        (tmp_path / "informe_2026-05-04.pdf").write_bytes(b"%PDF")
        result = audit_root_files(tmp_path)
        assert any("informe_2026-05-04.pdf" in v for v in result)

    def test_nul_file_flagged(self, tmp_path: Path) -> None:
        """'nul' file at root is flagged — skip on Windows (reserved device name)."""
        import sys

        if sys.platform == "win32":
            # 'nul' is a reserved Windows device name and cannot be created as a real file.
            # The actual 'nul' stray file in the repo is detected differently (pre-existing).
            pytest.skip("Cannot create 'nul' file on Windows (reserved device name)")
        (tmp_path / "nul").write_text("", encoding="utf-8")
        result = audit_root_files(tmp_path)
        assert any("nul" in v for v in result)

    def test_multiple_forbidden_files_all_returned(self, tmp_path: Path) -> None:
        """Multiple forbidden files are all in the result."""
        (tmp_path / "partidos_2026-05-04.json").write_text("{}", encoding="utf-8")
        (tmp_path / "features_dump.json").write_text("{}", encoding="utf-8")

        result = audit_root_files(tmp_path)
        assert len(result) == 2


class TestSubdirFilesNotFlagged:
    def test_forbidden_pattern_in_subdir_not_flagged(self, tmp_path: Path) -> None:
        """Files matching forbidden patterns in subdirs are NOT flagged."""
        subdir = tmp_path / "some_subdir"
        subdir.mkdir()
        (subdir / "partidos_2026-05-04.json").write_text("{}", encoding="utf-8")

        result = audit_root_files(tmp_path)
        assert result == []

    def test_runs_folder_contents_not_flagged(self, tmp_path: Path) -> None:
        """Files inside runs/ directory are not flagged."""
        runs = tmp_path / "runs" / "2026-05-04_j35-sabado"
        runs.mkdir(parents=True)
        (runs / "partidos_2026-05-04.json").write_text("{}", encoding="utf-8")

        result = audit_root_files(tmp_path)
        assert result == []


class TestMainBlock:
    def test_clean_exit_code_zero(self, tmp_path: Path) -> None:
        """__main__ exits with code 0 when no violations."""
        result = subprocess.run(
            [sys.executable, "-m", "audit.audit_root_files", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_violation_exit_code_nonzero(self, tmp_path: Path) -> None:
        """__main__ exits with non-zero code when violations found."""
        (tmp_path / "partidos_2026-05-04.json").write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "audit.audit_root_files", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
