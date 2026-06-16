"""
tests/overlay/test_batch_narratives.py — T7.5: Batch with --narratives dir.

Tests the _find_narrative_for_match helper that will look up a narrative YAML
for a given match in batch mode.  Matches with narratives get overlay applied;
matches without get normal prediction.

Tests:
  1. _find_narrative_for_match finds a YAML by home_vs_away_date naming.
  2. Returns None when no matching narrative exists.
  3. Non-narrative batch entries are predicted normally (no overlay applied).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the helper from run_prediction (T7.6 wires this)
# ---------------------------------------------------------------------------

try:
    from run_prediction import _find_narrative_for_match  # type: ignore[attr-defined]

    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindNarrativeForMatch:
    def test_finds_narrative_by_home_vs_away_date(self, tmp_path: Path) -> None:
        """_find_narrative_for_match finds a YAML file matching home_vs_away_date."""
        if not _IMPORT_OK:
            pytest.skip(
                "_find_narrative_for_match not yet implemented in run_prediction.py"
            )

        # Create a narrative file with the expected naming convention
        yaml_content = (
            "match:\n  home: Espanol\n  away: Levante\n  date: '2026-04-27'\n"
            "confidence_level: 3\n"
        )
        narrative_file = tmp_path / "Espanol_vs_Levante_2026-04-27.yaml"
        narrative_file.write_text(yaml_content, encoding="utf-8")

        result = _find_narrative_for_match(
            narratives_dir=tmp_path,
            home="Espanol",
            away="Levante",
            date="2026-04-27",
        )

        assert result is not None, "Expected to find the narrative file"
        assert result == narrative_file

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        """_find_narrative_for_match returns None when no file matches."""
        if not _IMPORT_OK:
            pytest.skip(
                "_find_narrative_for_match not yet implemented in run_prediction.py"
            )

        result = _find_narrative_for_match(
            narratives_dir=tmp_path,
            home="RealMadrid",
            away="Barcelona",
            date="2026-05-01",
        )

        assert result is None, f"Expected None but got: {result}"

    def test_returns_none_when_dir_does_not_exist(self, tmp_path: Path) -> None:
        """_find_narrative_for_match returns None when the directory doesn't exist."""
        if not _IMPORT_OK:
            pytest.skip(
                "_find_narrative_for_match not yet implemented in run_prediction.py"
            )

        missing_dir = tmp_path / "does_not_exist"
        result = _find_narrative_for_match(
            narratives_dir=missing_dir,
            home="Espanol",
            away="Levante",
            date="2026-04-27",
        )

        assert result is None
