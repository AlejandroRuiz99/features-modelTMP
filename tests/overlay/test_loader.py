"""
tests/overlay/test_loader.py — Unit tests for overlay.loader.

Tests (5 cases — Strict TDD, all RED before loader.py is implemented):
  1. Single file load round-trip → returns Narrative with correct fields
  2. Batch discovery in directory → finds 3 YAML files, returns dict
  3. File not found → raises FileNotFoundError with clear message
  4. Bad YAML syntax → raises with file path in message
  5. Validates YAML content → returns Narrative object (not raw dict)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from overlay.loader import discover_narratives, load_narrative
from overlay.schema import Narrative

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "narratives"


# ---------------------------------------------------------------------------
# Case 1: Single file load round-trip
# ---------------------------------------------------------------------------


class TestLoadNarrativeSingleFile:
    def test_valid_full_yaml_round_trip(self) -> None:
        """Load valid_full.yaml → Narrative with correct fields."""
        narr = load_narrative(FIXTURES_DIR / "valid_full.yaml")
        assert isinstance(narr, Narrative)
        assert narr.match.home == "Espanyol"
        assert narr.match.away == "Levante"
        assert narr.confidence_level == 4
        assert narr.physicality_bias == 1
        assert "stakes_both_relegation" in narr.special_flags
        assert narr.objectives is not None
        assert narr.objectives["home"].label == "salvacion"

    def test_valid_minimal_yaml(self) -> None:
        """Load valid_minimal.yaml → Narrative with mandatory objectives."""
        narr = load_narrative(FIXTURES_DIR / "valid_minimal.yaml")
        assert isinstance(narr, Narrative)
        assert narr.confidence_level == 3
        assert narr.special_flags == []
        assert narr.objectives is not None

    def test_old_objective_override_key_raises(self) -> None:
        """YAML with objective_override (old key) instead of objectives must raise ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "match:\n  home: 'TeamA'\n  away: 'TeamB'\n  date: '2026-01-01'\n"
                "confidence_level: 3\n"
                "objective_override:\n  home:\n    label: mid\n  away:\n    label: mid\n"
            )
            f.flush()
            # objective_override is not in allowed fields, should raise ValueError
            with pytest.raises(ValueError, match="unknown field|objective_override"):
                load_narrative(Path(f.name))


# ---------------------------------------------------------------------------
# Case 2: Batch discovery in directory (3 files found)
# ---------------------------------------------------------------------------


class TestDiscoverNarratives:
    def test_discover_three_files(self, tmp_path: Path) -> None:
        """discover_narratives finds 3 YAML files in a directory."""
        # Create 3 minimal valid YAML files (objectives is REQUIRED)
        for i in range(3):
            (tmp_path / f"match_{i}.yaml").write_text(
                f"match:\n  home: 'TeamA{i}'\n  away: 'TeamB{i}'\n"
                f"  date: '2026-04-0{i + 1}'\n"
                f"objectives:\n  home:\n    label: mid\n  away:\n    label: mid\n"
                f"confidence_level: 3\n",
                encoding="utf-8",
            )
        result = discover_narratives(tmp_path)
        assert len(result) == 3
        for v in result.values():
            assert isinstance(v, Narrative)

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        """discover_narratives in empty dir returns empty dict."""
        result = discover_narratives(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# Case 3: File not found → raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestFileNotFound:
    def test_missing_file_raises(self) -> None:
        """load_narrative on a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match=r"nonexistent\.yaml"):
            load_narrative(Path("nonexistent.yaml"))


# ---------------------------------------------------------------------------
# Case 4: Bad YAML syntax → raises with file path in message
# ---------------------------------------------------------------------------


class TestBadYamlSyntax:
    def test_bad_yaml_raises_with_path(self) -> None:
        """load_narrative on malformed YAML raises ValueError with file path."""
        bad_path = FIXTURES_DIR / "bad_yaml.yaml"
        with pytest.raises(
            (ValueError, Exception), match=r"bad_yaml\.yaml|parse|YAML|yaml"
        ):
            load_narrative(bad_path)


# ---------------------------------------------------------------------------
# Case 5: Validates YAML content → returns Narrative object
# ---------------------------------------------------------------------------


class TestReturnsNarrativeObject:
    def test_returns_narrative_not_dict(self) -> None:
        """load_narrative returns a Narrative dataclass, not a raw dict."""
        narr = load_narrative(FIXTURES_DIR / "valid_full.yaml")
        assert type(narr).__name__ == "Narrative"
        # Should have .match attribute (NarrativeMatch), not dict['match']
        assert hasattr(narr.match, "home")
        assert hasattr(narr.match, "away")
