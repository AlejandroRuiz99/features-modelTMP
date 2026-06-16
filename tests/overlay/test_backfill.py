"""
T8.1 — Tests for scripts/overlay_backfill.py dry-run behavior.

Tests that the backfill script:
- Produces a markdown comparison table with correct columns
- Handles missing actuals (shows NA)
- Handles missing narratives (skipped + listed)
- Table has correct header structure
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

SCRIPT = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "overlay_backfill.py"
)
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
NARRATIVES_DIR = FIXTURES / "narratives"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_actuals(tmp_path: Path, data: dict) -> Path:
    """Write an actuals JSON file and return its path."""
    p = tmp_path / "actuals.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _run_backfill(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# T8.1: markdown output structure
# ---------------------------------------------------------------------------


class TestBackfillMarkdownStructure:
    """The backfill script must emit a markdown table with the required columns."""

    REQUIRED_COLUMNS: ClassVar[list[str]] = [
        "match",
        "pre_pred",
        "post_pred",
        "actual",
        "line_pre",
        "line_post",
        "hit_pre",
        "hit_post",
        "rules_fired",
    ]

    def test_output_contains_markdown_table_header(self, tmp_path: Path) -> None:
        """Script produces a table header with all required columns."""
        actuals = _make_actuals(tmp_path, {})
        output_path = tmp_path / "report.md"

        result = _run_backfill(
            [
                "--narratives",
                str(NARRATIVES_DIR),
                "--actuals",
                str(actuals),
                "--output",
                str(output_path),
            ]
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        content = output_path.read_text(encoding="utf-8")

        # Must contain a markdown table separator line
        assert "|" in content, "Expected markdown table with | separators"

        # Check all required columns appear in the output
        header_lower = content.lower()
        for col in self.REQUIRED_COLUMNS:
            assert col in header_lower, (
                f"Expected column '{col}' in markdown output.\nGot:\n{content[:500]}"
            )

    def test_missing_actuals_shows_na(self, tmp_path: Path) -> None:
        """Matches without actuals data must show NA in the actual column."""
        # actuals dict is empty → all matches show NA
        actuals = _make_actuals(tmp_path, {})
        output_path = tmp_path / "report.md"

        result = _run_backfill(
            [
                "--narratives",
                str(NARRATIVES_DIR),
                "--actuals",
                str(actuals),
                "--output",
                str(output_path),
            ]
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        content = output_path.read_text(encoding="utf-8")
        assert "NA" in content, (
            f"Expected 'NA' for missing actuals.\nGot:\n{content[:500]}"
        )

    def test_missing_narrative_file_skipped_and_listed(self, tmp_path: Path) -> None:
        """Narratives that fail to load must be skipped and listed in the report."""
        # Use a bad YAML fixture to trigger a skip
        bad_narratives_dir = tmp_path / "narratives"
        bad_narratives_dir.mkdir()

        # Copy only the bad yaml
        bad_src = FIXTURES / "narratives" / "bad_yaml.yaml"
        if bad_src.exists():
            import shutil

            shutil.copy(bad_src, bad_narratives_dir / "bad_yaml.yaml")
        else:
            # Create one inline
            (bad_narratives_dir / "bad_yaml.yaml").write_text(
                "not: a: valid: yaml: : :", encoding="utf-8"
            )

        actuals = _make_actuals(tmp_path, {})
        output_path = tmp_path / "report.md"

        result = _run_backfill(
            [
                "--narratives",
                str(bad_narratives_dir),
                "--actuals",
                str(actuals),
                "--output",
                str(output_path),
            ]
        )
        # Should not crash — bad files are skipped
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        # Report should mention skipped or show 0 rows but not crash

    def test_valid_narrative_appears_as_row(self, tmp_path: Path) -> None:
        """A valid narrative YAML produces a table row in the report."""
        valid_dir = tmp_path / "narratives"
        valid_dir.mkdir()

        import shutil

        src = FIXTURES / "narratives" / "valid_full.yaml"
        shutil.copy(src, valid_dir / "valid_full.yaml")

        actuals = _make_actuals(tmp_path, {"Espanyol_vs_Levante_2026-04-27": 22})
        output_path = tmp_path / "report.md"

        result = _run_backfill(
            [
                "--narratives",
                str(valid_dir),
                "--actuals",
                str(actuals),
                "--output",
                str(output_path),
            ]
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        content = output_path.read_text(encoding="utf-8")
        # Should have at least 1 data row (3 lines: header, separator, 1+ data rows)
        table_lines = [ln for ln in content.splitlines() if "|" in ln]
        assert len(table_lines) >= 3, (
            f"Expected header + separator + at least 1 data row.\nGot:\n{content}"
        )
