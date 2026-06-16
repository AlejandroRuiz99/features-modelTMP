"""
Unit tests for predecir_jornada calendar parser (R1).

Tests written FIRST (TDD RED phase) — implementation in predecir_jornada.py
does not exist yet, so all tests will fail with ImportError.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest

# Import the module under test — will fail until implementation exists
from predecir_jornada import CalendarioPartido, parse_calendario

MINIMAL_CALENDAR = textwrap.dedent("""\
    ================================================================================
                        CALENDARIO LA LIGA - PRIMERA DIVISIÓN
    ================================================================================

    ################################################################################
                             TEMPORADA 2025-2026
    ################################################################################

    --- Jornada 1 (17/08/2025) ---
    Real Madrid - Ath Madrid
    Barcelona - Celta
    """)

TWO_SEASON_CALENDAR = textwrap.dedent("""\
    ################################################################################
                             TEMPORADA 2024-2025
    ################################################################################

    --- Jornada 1 (18/08/2024) ---
    Real Madrid - Ath Bilbao
    Barcelona - Vallecano

    ################################################################################
                             TEMPORADA 2025-2026
    ################################################################################

    --- Jornada 1 (17/08/2025) ---
    Ath Madrid - Getafe
    Sevilla - Osasuna
    """)

MALFORMED_LINE_CALENDAR = textwrap.dedent("""\
    ################################################################################
                             TEMPORADA 2025-2026
    ################################################################################

    --- Jornada 1 (17/08/2025) ---
    Real Madrid - Ath Madrid
    !!!INVALID_LINE!!!
    Barcelona - Celta
    """)

SPECIAL_NAMES_CALENDAR = textwrap.dedent("""\
    ################################################################################
                             TEMPORADA 2025-2026
    ################################################################################

    --- Jornada 10 (19/10/2025) ---
    Atlético de Madrid - Athletic Club
    Alavés - Cádiz
    """)


class TestParseCalendarioHappyPath:
    """R1: Happy path — well-formed calendar returns all matches."""

    def test_returns_list_of_calendario_partido(self) -> None:
        """Happy path: parser returns list of CalendarioPartido instances."""
        partidos = parse_calendario(text=MINIMAL_CALENDAR)
        assert isinstance(partidos, list)
        assert len(partidos) == 2
        assert all(isinstance(p, CalendarioPartido) for p in partidos)

    def test_match_fields_populated(self) -> None:
        """Each partido has local, visitante, nominal_date, jornada, temporada."""
        partidos = parse_calendario(text=MINIMAL_CALENDAR)
        p = partidos[0]
        assert p.local == "Real Madrid"
        assert p.visitante == "Ath Madrid"
        assert isinstance(p.nominal_date, date)
        assert p.jornada == 1
        assert p.temporada == "2025-2026"


class TestParseCalendarioMultiSeason:
    """R1: Multi-season — each match carries correct temporada."""

    def test_each_match_has_correct_season(self) -> None:
        """Matches in different season blocks have correct temporada."""
        partidos = parse_calendario(text=TWO_SEASON_CALENDAR)
        season_2024 = [p for p in partidos if p.temporada == "2024-2025"]
        season_2025 = [p for p in partidos if p.temporada == "2025-2026"]
        assert len(season_2024) == 2
        assert len(season_2025) == 2

    def test_total_match_count(self) -> None:
        """All 4 matches are returned across 2 seasons."""
        partidos = parse_calendario(text=TWO_SEASON_CALENDAR)
        assert len(partidos) == 4


class TestParseCalendarioMalformedLine:
    """R1: Malformed line is skipped with warning, others returned."""

    def test_malformed_line_skipped(self) -> None:
        """A syntactically invalid match line is skipped."""
        partidos = parse_calendario(text=MALFORMED_LINE_CALENDAR)
        # 3 lines total: Real Madrid - Ath Madrid, INVALID, Barcelona - Celta
        # Only 2 valid lines returned
        assert len(partidos) == 2

    def test_valid_lines_still_returned(self) -> None:
        """Valid lines around malformed line are parsed."""
        partidos = parse_calendario(text=MALFORMED_LINE_CALENDAR)
        teams = [(p.local, p.visitante) for p in partidos]
        assert ("Real Madrid", "Ath Madrid") in teams
        assert ("Barcelona", "Celta") in teams


class TestParseCalendarioFileMissing:
    """R1: Missing file raises descriptive error."""

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """parse_calendario(path=...) raises descriptive error when file missing."""
        missing = tmp_path / "nonexistent_calendario.txt"
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            parse_calendario(path=missing)


class TestParseCalendarioDateParsing:
    """R1: Date parsing — DD/MM/YYYY format."""

    def test_date_12_04_2026(self) -> None:
        """Match line with date 12/04/2026 → nominal_date = date(2026, 4, 12)."""
        cal = textwrap.dedent("""\
            ################################################################################
                                     TEMPORADA 2025-2026
            ################################################################################

            --- Jornada 31 (12/04/2026) ---
            Sociedad - Sevilla
            """)
        partidos = parse_calendario(text=cal)
        assert partidos[0].nominal_date == date(2026, 4, 12)


class TestParseCalendarioSpecialTeamNames:
    """R1: Special team names normalized via team_mapping."""

    def test_atletico_normalized(self) -> None:
        """'Atlético de Madrid' → 'Ath Madrid' after normalize()."""
        partidos = parse_calendario(text=SPECIAL_NAMES_CALENDAR)
        teams_local = [p.local for p in partidos]
        assert "Ath Madrid" in teams_local

    def test_athletic_club_normalized(self) -> None:
        """'Athletic Club' → 'Ath Bilbao' after normalize()."""
        partidos = parse_calendario(text=SPECIAL_NAMES_CALENDAR)
        teams_visit = [p.visitante for p in partidos]
        assert "Ath Bilbao" in teams_visit

    def test_alaves_normalized(self) -> None:
        """'Alavés' → 'Alaves' after normalize()."""
        partidos = parse_calendario(text=SPECIAL_NAMES_CALENDAR)
        teams_local = [p.local for p in partidos]
        assert "Alaves" in teams_local


class TestParseCalendarioEncoding:
    """R1: UTF-8 encoding — accents handled correctly."""

    def test_accented_chars_in_team_names(self) -> None:
        """Team names with accents parsed without encoding errors."""
        cal = textwrap.dedent("""\
            ################################################################################
                                     TEMPORADA 2025-2026
            ################################################################################

            --- Jornada 5 (14/09/2025) ---
            Cádiz - Leganés
            """)
        # Should not raise any UnicodeDecodeError
        partidos = parse_calendario(text=cal)
        # Cádiz -> Cadiz, Leganés -> Leganes via normalize
        assert len(partidos) == 1

    def test_em_dash_separator(self) -> None:
        """Match line with en-dash separator (U+2013) is parsed correctly."""
        cal = textwrap.dedent("""\
            ################################################################################
                                     TEMPORADA 2025-2026
            ################################################################################

            --- Jornada 5 (14/09/2025) ---
            Real Madrid \u2013 Barcelona
            """)
        partidos = parse_calendario(text=cal)
        assert len(partidos) == 1
        assert partidos[0].local == "Real Madrid"
        assert partidos[0].visitante == "Barcelona"


class TestParseCalendarioEmptyFile:
    """R1: Empty file returns empty list."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty calendar text returns empty list."""
        partidos = parse_calendario(text="")
        assert partidos == []
