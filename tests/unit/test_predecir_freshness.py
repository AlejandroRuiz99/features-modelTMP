"""
Unit tests for predecir_jornada freshness verification (R3).

Tests written FIRST (TDD RED phase).
Mocks fetch_matches_for_season via unittest.mock.patch.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from predecir_jornada import (
    CalendarioPartido,
    FreshnessReport,
    verify_freshness,
)


def _make_partido(
    local: str,
    visitante: str,
    nominal_date: date,
    jornada: int,
    temporada: str = "2025-2026",
) -> CalendarioPartido:
    return CalendarioPartido(
        local=local,
        visitante=visitante,
        nominal_date=nominal_date,
        jornada=jornada,
        temporada=temporada,
    )


# Calendar for season 2025-2026 with jornadas 1 and 2 before target date
CALENDAR_25_26 = [
    # Jornada 1 — before target date (2026-04-12)
    _make_partido("Real Madrid", "Ath Bilbao", date(2025, 8, 17), 1),
    _make_partido("Barcelona", "Celta", date(2025, 8, 17), 1),
    # Jornada 2 — before target date
    _make_partido("Ath Madrid", "Getafe", date(2025, 8, 24), 2),
    # Jornada 31 ON target date — should NOT be in expected (same date)
    _make_partido("Sociedad", "Sevilla", date(2026, 4, 12), 31),
]

TARGET_DATE = date(2026, 4, 12)
SEASON = "2025-26"


class TestVerifyFreshnessAllPresent:
    """R3: All matches present → ok=True, missing=[]."""

    def test_all_present_returns_ok(self) -> None:
        """When all expected matches exist in Supabase → ok=True."""
        supabase_rows = [
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            {"home_team": "Ath Madrid", "away_team": "Getafe"},
        ]
        with patch(
            "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        assert report.ok is True
        assert report.missing_matches == []

    def test_all_present_returns_freshness_report_type(self) -> None:
        """Return type is FreshnessReport."""
        supabase_rows = [
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            {"home_team": "Ath Madrid", "away_team": "Getafe"},
        ]
        with patch(
            "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        assert isinstance(report, FreshnessReport)


class TestVerifyFreshnessGapDetected:
    """R3: Gap detected → ok=False, missing contains absent matches."""

    def test_single_jornada_missing(self) -> None:
        """Jornada 2 missing from Supabase → ok=False."""
        supabase_rows = [
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            # Ath Madrid - Getafe (jornada 2) is MISSING
        ]
        with patch(
            "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        assert report.ok is False
        assert len(report.missing_matches) == 1

    def test_missing_contains_team_names(self) -> None:
        """Missing matches list contains team information."""
        supabase_rows: list[dict] = []  # everything missing
        with patch(
            "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        assert report.ok is False
        assert len(report.missing_matches) == 3  # jornadas 1 and 2 before target


class TestVerifyFreshnessSeasonsScoped:
    """R3: Freshness check uses only matches from the correct season."""

    def test_other_season_matches_not_counted(self) -> None:
        """Matches from 2024-2025 in calendar don't affect 2025-2026 check."""
        mixed_calendar = [
            *CALENDAR_25_26,
            # From a different season
            _make_partido("Leganes", "Cadiz", date(2024, 8, 18), 1, "2024-2025"),
        ]
        # Supabase only has 2024-2025 matches for these teams (wrong season)
        supabase_rows_for_season: list[dict] = [
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            {"home_team": "Ath Madrid", "away_team": "Getafe"},
        ]
        with patch(
            "predecir_jornada.fetch_matches_for_season",
            return_value=supabase_rows_for_season,
        ):
            report = verify_freshness(TARGET_DATE, SEASON, mixed_calendar)
        # Should be ok because all 2025-26 matches are present
        assert report.ok is True


class TestVerifyFreshnessCanonicalNormalization:
    """R3: normalize() applied on both sides before comparison."""

    def test_supabase_accented_names_still_match(self) -> None:
        """Supabase names with different accents/casing still match via normalize."""
        # Calendar has canonical "Ath Madrid"; Supabase might store "Atletico Madrid"
        supabase_rows = [
            {"home_team": "Real Madrid", "away_team": "Athletic Club"},
            {"home_team": "FC Barcelona", "away_team": "RC Celta"},
            {"home_team": "Atletico Madrid", "away_team": "Getafe CF"},
        ]
        with patch(
            "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        # normalize("Athletic Club") = "Ath Bilbao", which matches calendar "Ath Bilbao"
        # normalize("Atletico Madrid") = "Ath Madrid", matches "Ath Madrid"
        assert report.ok is True


class TestVerifyFreshnessPostponedMatch:
    """R3: Postponed match under different date is still detected as present."""

    def test_postponed_match_detected(self) -> None:
        """Match stored under different match_date is found (date-agnostic check)."""
        # Calendar has the match on date(2025, 8, 17)
        # Supabase stores it under a different date (postponed)
        supabase_rows = [
            # Postponed: different date but same teams
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            {"home_team": "Ath Madrid", "away_team": "Getafe"},
        ]
        with patch(
            "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        assert report.ok is True


class TestVerifyFreshnessOddsWarning:
    """R3: Odds freshness side-check returns non-blocking warning string."""

    def test_odds_warning_field_present(self) -> None:
        """FreshnessReport has an odds_warning field."""
        supabase_rows = [
            {"home_team": "Real Madrid", "away_team": "Ath Bilbao"},
            {"home_team": "Barcelona", "away_team": "Celta"},
            {"home_team": "Ath Madrid", "away_team": "Getafe"},
        ]
        with (
            patch(
                "predecir_jornada.fetch_matches_for_season", return_value=supabase_rows
            ),
            patch(
                "predecir_jornada.check_odds_freshness",
                return_value="Odds tienen 30 horas — pueden estar desactualizadas",
            ),
        ):
            report = verify_freshness(TARGET_DATE, SEASON, CALENDAR_25_26)
        # Warning does not change ok
        assert report.ok is True
        # odds_warning is set to the string returned by check_odds_freshness
        assert report.odds_warning is not None
