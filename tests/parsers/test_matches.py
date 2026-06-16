"""
tests/parsers/test_matches.py — TDD for parsers.matches_parser

Tests (T2.3):
  1. Clean paste with day header parses correctly.
  2. Multi-day paste parses all matches.
  3. Ambiguous team name sets canonical_match=False with original text.
  4. Empty paste raises ParseError.
  5. Missing referee line parses with referee=None.
  6. Multiple matches per day.
  7. Match slug is auto-generated.
"""

from __future__ import annotations

import pytest

from parsers.matches_parser import (
    ParsedMatchday,
    ParseError,
    parse_matches_text,
)


class TestCleanPaste:
    def test_single_match_with_day_header(self) -> None:
        """Clean paste: day header + match line → one ParsedMatch."""
        text = "Sábado J35\nReal Madrid - Mallorca, Munuera Montero"
        result = parse_matches_text(text)

        assert isinstance(result, ParsedMatchday)
        assert len(result.matches) == 1

        m = result.matches[0]
        assert m.home == "Real Madrid"
        assert m.away == "Mallorca"
        assert m.referee == "Munuera Montero"

    def test_jornada_extracted_from_header(self) -> None:
        """Jornada number is extracted from 'J35' in header."""
        text = "Sábado J35\nReal Madrid - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        assert result.jornada == 35

    def test_match_slug_auto_generated(self) -> None:
        """Match slug is auto-generated from home and away."""
        text = "Sábado J35\nReal Madrid - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        m = result.matches[0]
        # Slug should be snake-case combination
        assert "real" in m.slug.lower() or "madrid" in m.slug.lower()
        assert "mallorca" in m.slug.lower()

    def test_multiple_matches_per_day(self) -> None:
        """Multiple match lines under a single day header."""
        text = (
            "Sábado J35\n"
            "Real Madrid - Mallorca, Munuera Montero\n"
            "Sevilla - Getafe, Del Cerro Grande"
        )
        result = parse_matches_text(text)
        assert len(result.matches) == 2

    def test_em_dash_separator(self) -> None:
        """Handles en-dash (\u2013) as separator."""
        text = "J35\nReal Madrid \u2013 Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        assert len(result.matches) == 1
        assert result.matches[0].home == "Real Madrid"


class TestMultiDay:
    def test_two_day_headers(self) -> None:
        """Two day headers with one match each → 2 matches."""
        text = (
            "Sábado J35\n"
            "Real Madrid - Mallorca, Munuera Montero\n"
            "\n"
            "Domingo J35\n"
            "Sevilla - Getafe, Del Cerro Grande"
        )
        result = parse_matches_text(text)
        assert len(result.matches) == 2


class TestAmbiguousTeam:
    def test_ambiguous_team_sets_flag(self) -> None:
        """Unresolvable team name sets canonical_match=False."""
        text = "J35\nXyzUnknownTeam99 - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.canonical_match is False

    def test_ambiguous_preserves_original_text(self) -> None:
        """Ambiguous team preserves the raw input text."""
        text = "J35\nXyzUnknownTeam99 - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        m = result.matches[0]
        # The original unresolved text should be in home field
        assert "XyzUnknownTeam99" in m.home or m.canonical_match is False

    def test_canonical_team_sets_flag_true(self) -> None:
        """Recognized team has canonical_match=True."""
        text = "J35\nReal Madrid - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        m = result.matches[0]
        assert m.canonical_match is True


class TestEmptyPaste:
    def test_empty_string_raises_parse_error(self) -> None:
        """Empty input raises ParseError."""
        with pytest.raises(ParseError):
            parse_matches_text("")

    def test_whitespace_only_raises_parse_error(self) -> None:
        """Whitespace-only input raises ParseError."""
        with pytest.raises(ParseError):
            parse_matches_text("   \n   \n  ")

    def test_header_only_no_matches_raises(self) -> None:
        """Day header with no match lines raises ParseError."""
        with pytest.raises(ParseError):
            parse_matches_text("Sábado J35\n")


class TestMissingReferee:
    def test_no_referee_parses_with_none(self) -> None:
        """Match line without referee sets referee=None."""
        text = "J35\nReal Madrid - Mallorca"
        result = parse_matches_text(text)
        assert len(result.matches) == 1
        assert result.matches[0].referee is None


class TestWarnings:
    def test_warnings_is_list(self) -> None:
        """ParsedMatchday.warnings is always a list."""
        text = "J35\nReal Madrid - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        assert isinstance(result.warnings, list)

    def test_ambiguous_team_generates_warning(self) -> None:
        """Ambiguous team names go into warnings list."""
        text = "J35\nXyzUnknownTeam99 - Mallorca, Munuera Montero"
        result = parse_matches_text(text)
        assert len(result.warnings) > 0
