"""
tests/parsers/test_narrative_parser.py — TDD for parsers.narrative_parser

Tests (T2.5):
  1. NARRATIVE_TEMPLATE is a non-empty string with required placeholders.
  2. validate_narrative_yaml passes a valid YAML file (delegates to overlay/loader.py).
  3. validate_narrative_yaml raises on missing objectives field.
  4. narrative_cached() returns Path when file exists and is non-empty.
  5. narrative_cached() returns None when file does not exist.
  6. narrative_cached() returns None when file is empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parsers.narrative_parser import (
    NARRATIVE_PROMPT_TEMPLATE,
    narrative_cached,
    validate_narrative_yaml,
)

# ---------------------------------------------------------------------------
# Minimal valid narrative YAML for testing
# ---------------------------------------------------------------------------

_VALID_NARRATIVE_YAML = """
match:
  home: "Real Madrid"
  away: "Mallorca"
  date: "2026-05-04"
  jornada: 35

objectives:
  home:
    label: titulo
  away:
    label: mid

confidence_level: 4
"""

_NARRATIVE_MISSING_CONFIDENCE = """
match:
  home: "Real Madrid"
  away: "Mallorca"
  date: "2026-05-04"
"""


class TestNarrativeTemplate:
    def test_template_is_non_empty_string(self) -> None:
        """NARRATIVE_PROMPT_TEMPLATE is a non-empty string."""
        assert isinstance(NARRATIVE_PROMPT_TEMPLATE, str)
        assert len(NARRATIVE_PROMPT_TEMPLATE) > 50

    def test_template_contains_home_placeholder(self) -> None:
        """Template contains {home} placeholder."""
        assert "{home}" in NARRATIVE_PROMPT_TEMPLATE

    def test_template_contains_away_placeholder(self) -> None:
        """Template contains {away} placeholder."""
        assert "{away}" in NARRATIVE_PROMPT_TEMPLATE

    def test_template_contains_date_placeholder(self) -> None:
        """Template contains {date} placeholder."""
        assert "{date}" in NARRATIVE_PROMPT_TEMPLATE

    def test_template_mentions_confidence_level(self) -> None:
        """Template mentions confidence_level (so LLM knows to fill it)."""
        assert "confidence_level" in NARRATIVE_PROMPT_TEMPLATE.lower()


class TestValidateNarrativeYaml:
    def test_valid_yaml_returns_narrative(self, tmp_path: Path) -> None:
        """validate_narrative_yaml returns a Narrative for valid YAML."""
        f = tmp_path / "match.yaml"
        f.write_text(_VALID_NARRATIVE_YAML, encoding="utf-8")

        narrative = validate_narrative_yaml(f)
        assert narrative is not None
        assert narrative.match.home == "Real Madrid"
        assert narrative.match.away == "Mallorca"

    def test_missing_confidence_level_raises(self, tmp_path: Path) -> None:
        """validate_narrative_yaml raises ValueError if confidence_level is missing."""
        f = tmp_path / "match.yaml"
        f.write_text(_NARRATIVE_MISSING_CONFIDENCE, encoding="utf-8")

        with pytest.raises(ValueError):
            validate_narrative_yaml(f)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """validate_narrative_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            validate_narrative_yaml(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """validate_narrative_yaml raises ValueError for broken YAML."""
        f = tmp_path / "match.yaml"
        f.write_text("not: valid: yaml: [unclosed", encoding="utf-8")

        with pytest.raises(ValueError):
            validate_narrative_yaml(f)


class TestNarrativeCached:
    def test_returns_path_when_file_exists(self, tmp_path: Path) -> None:
        """narrative_cached() returns Path if the narrative file exists and is non-empty."""
        narr_dir = tmp_path / "narratives"
        narr_dir.mkdir()
        slug = "realmadrid_vs_mallorca_2026-05-04"
        narr_file = narr_dir / f"{slug}.yaml"
        narr_file.write_text(_VALID_NARRATIVE_YAML, encoding="utf-8")

        result = narrative_cached(slug, narr_dir)

        assert result is not None
        assert result == narr_file

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """narrative_cached() returns None if file does not exist."""
        narr_dir = tmp_path / "narratives"
        narr_dir.mkdir()
        slug = "nonexistent_match"

        result = narrative_cached(slug, narr_dir)

        assert result is None

    def test_returns_none_when_file_empty(self, tmp_path: Path) -> None:
        """narrative_cached() returns None if file exists but is empty."""
        narr_dir = tmp_path / "narratives"
        narr_dir.mkdir()
        slug = "empty_match"
        (narr_dir / f"{slug}.yaml").write_text("", encoding="utf-8")

        result = narrative_cached(slug, narr_dir)

        assert result is None

    def test_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        """narrative_cached() returns None if narratives dir doesn't exist."""
        narr_dir = tmp_path / "nonexistent_dir"
        result = narrative_cached("some_match", narr_dir)
        assert result is None
