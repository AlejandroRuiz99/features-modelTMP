"""
Unit tests for predecir_jornada filter_matches (R2).

Tests written FIRST (TDD RED phase).
"""

from __future__ import annotations

from datetime import date

import pytest
from predecir_jornada import CalendarioPartido, filter_matches


def _make_partido(
    local: str,
    visitante: str,
    nominal_date: date,
    jornada: int,
    temporada: str,
) -> CalendarioPartido:
    return CalendarioPartido(
        local=local,
        visitante=visitante,
        nominal_date=nominal_date,
        jornada=jornada,
        temporada=temporada,
    )


# ---- Fixture: 3 seasons, multiple jornadas ----
PARTIDOS: list[CalendarioPartido] = [
    # 2024-2025 — Jornada 1
    _make_partido("Real Madrid", "Ath Bilbao", date(2024, 8, 18), 1, "2024-2025"),
    _make_partido("Barcelona", "Vallecano", date(2024, 8, 18), 1, "2024-2025"),
    # 2024-2025 — Jornada 10
    _make_partido("Getafe", "Sevilla", date(2024, 10, 20), 10, "2024-2025"),
    # 2025-2026 — Jornada 1
    _make_partido("Ath Madrid", "Getafe", date(2025, 8, 17), 1, "2025-2026"),
    _make_partido("Sevilla", "Osasuna", date(2025, 8, 17), 1, "2025-2026"),
    # 2025-2026 — Jornada 31 (12/04/2026)
    _make_partido("Sociedad", "Leganes", date(2026, 4, 12), 31, "2025-2026"),
    _make_partido("Barcelona", "Ath Madrid", date(2026, 4, 12), 31, "2025-2026"),
    # 2025-2026 — Jornada 31 (13/04/2026, different date)
    _make_partido("Real Madrid", "Vallecano", date(2026, 4, 13), 31, "2025-2026"),
]


class TestFilterByDate:
    """R2: filter by target_date."""

    def test_filter_returns_matches_on_date(self) -> None:
        """Matches with nominal_date == target_date are returned."""
        result = filter_matches(PARTIDOS, target_date=date(2026, 4, 12))
        assert len(result) == 2
        dates = {p.nominal_date for p in result}
        assert dates == {date(2026, 4, 12)}

    def test_filter_by_date_correct_teams(self) -> None:
        """Correct team pairs returned for the target date."""
        result = filter_matches(PARTIDOS, target_date=date(2026, 4, 12))
        teams = {(p.local, p.visitante) for p in result}
        assert ("Sociedad", "Leganes") in teams
        assert ("Barcelona", "Ath Madrid") in teams

    def test_filter_by_date_no_results(self) -> None:
        """Date with no matches returns empty list."""
        result = filter_matches(PARTIDOS, target_date=date(2099, 1, 1))
        assert result == []


class TestFilterByMatchdaySeason:
    """R2: filter by jornada + temporada."""

    def test_filter_by_jornada_and_season(self) -> None:
        """All matches of jornada 1 in 2024-2025 returned."""
        result = filter_matches(PARTIDOS, jornada=1, temporada="2024-2025")
        assert len(result) == 2
        assert all(p.temporada == "2024-2025" for p in result)
        assert all(p.jornada == 1 for p in result)

    def test_filter_jornada_31_2025_2026(self) -> None:
        """Jornada 31 in 2025-2026 returns 3 matches (spanning 2 dates)."""
        result = filter_matches(PARTIDOS, jornada=31, temporada="2025-2026")
        assert len(result) == 3
        assert all(p.jornada == 31 for p in result)


class TestFilterAmbiguousMatchday:
    """R2: jornada without season when multiple seasons exist → error."""

    def test_ambiguous_jornada_raises(self) -> None:
        """Jornada 1 without season → ValueError or AmbiguousMatchdayError."""
        with pytest.raises((ValueError, KeyError)):
            filter_matches(PARTIDOS, jornada=1)

    def test_unambiguous_jornada_ok_when_only_one_season(self) -> None:
        """Jornada without season OK when only one season in list."""
        single_season = [p for p in PARTIDOS if p.temporada == "2025-2026"]
        result = filter_matches(single_season, jornada=1)
        assert len(result) == 2


class TestFilterSeasonAutoDetect:
    """R2: Season auto-detected from target_date."""

    def test_date_in_2025_2026_season(self) -> None:
        """target_date=2026-04-12 belongs to season 2025-2026."""
        result = filter_matches(PARTIDOS, target_date=date(2026, 4, 12))
        # All results should be from 2025-2026 (the auto-detected season)
        assert all(p.temporada == "2025-2026" for p in result)

    def test_date_2024_2025_season(self) -> None:
        """target_date=2024-10-20 belongs to season 2024-2025."""
        result = filter_matches(PARTIDOS, target_date=date(2024, 10, 20))
        assert all(p.temporada == "2024-2025" for p in result)
