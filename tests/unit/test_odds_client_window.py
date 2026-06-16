"""Tests for odds_client.get_match_odds_rows() with the new match_date window.

Regression test for the J30 2025-26 bug where only the latest scrape of
odds_raw was queried, causing 9/10 matches to be missed because the
Codere scraper runs daily and different matches appear in different
daily scrapes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from selection import odds_client


# Shared fixture data: a mix of rows from different scraped_at dates
# representing J30 2025-26.
_J30_ROWS = [
    # Thursday scrape — Vallecano vs Elche (Codere uses "Elche vs Rayo")
    {
        "home_team": "Elche",
        "away_team": "Rayo",
        "mercado": "Total de Faltas Cometidas Más/Menos",
        "selection": "Más de 25.5",
        "cuota": 1.72,
        "external_event_id": "ev_rayo_elche",
        "partido": "Elche vs Rayo",
        "sport": "soccer",
        "scraped_at": "2026-04-03T09:04:05+00:00",
    },
    {
        "home_team": "Elche",
        "away_team": "Rayo",
        "mercado": "Total de Faltas Cometidas Más/Menos",
        "selection": "Menos de 25.5",
        "cuota": 2.10,
        "external_event_id": "ev_rayo_elche",
        "partido": "Elche vs Rayo",
        "sport": "soccer",
        "scraped_at": "2026-04-03T09:04:05+00:00",
    },
    # Saturday scrape — Mallorca vs Real Madrid
    {
        "home_team": "Mallorca",
        "away_team": "Real Madrid",
        "mercado": "Total de Faltas Cometidas Más/Menos",
        "selection": "Más de 22.5",
        "cuota": 1.80,
        "external_event_id": "ev_mallorca_rm",
        "partido": "Mallorca vs Real Madrid",
        "sport": "soccer",
        "scraped_at": "2026-04-04T09:02:45+00:00",
    },
    # Monday scrape — Girona vs Villarreal (latest overall)
    {
        "home_team": "Girona",
        "away_team": "Villarreal",
        "mercado": "Total de Faltas Cometidas Más/Menos",
        "selection": "Más de 24.5",
        "cuota": 1.82,
        "external_event_id": "ev_girona_villa",
        "partido": "Girona vs Villarreal",
        "sport": "soccer",
        "scraped_at": "2026-04-06T09:01:16+00:00",
    },
]


# Minimal scores dict — odds_client.resolve() uses buscar_equipo() which
# falls back to the raw name when the team is not in scores.
_EMPTY_SCORES: dict = {}

# Populated scores dict used when we need fuzzy_name_search to resolve
# short forms like 'Athletic' / 'Rayo' to canonical names via TEAM_ALIASES.
# buscar_equipo() requires the canonical name to be present as a candidate
# (i.e. as a key in scores) to return it, so we populate the J30 canonical
# team names with empty stat dicts.
_J30_SCORES: dict = {
    "Alaves": {},
    "Osasuna": {},
    "Ath Madrid": {},
    "Barcelona": {},
    "Betis": {},
    "Espanol": {},
    "Girona": {},
    "Villarreal": {},
    "Mallorca": {},
    "Real Madrid": {},
    "Oviedo": {},
    "Sevilla": {},
    "Getafe": {},
    "Ath Bilbao": {},
    "Valencia": {},
    "Celta": {},
    "Vallecano": {},
    "Elche": {},
    "Sociedad": {},
    "Levante": {},
}


class TestGetMatchOddsRowsLegacyBehavior:
    """When match_date is None, behavior must stay identical to before."""

    def test_no_match_date_uses_fetch_latest(self) -> None:
        """With match_date=None, only the latest scrape is queried."""
        with (
            patch.object(
                odds_client.supabase_client,
                "fetch_latest_odds_rows",
                return_value={
                    "scraped_at": "2026-04-06T09:01:16+00:00",
                    "rows": [_J30_ROWS[3]],  # only Girona vs Villarreal
                },
            ) as mock_latest,
            patch.object(
                odds_client.supabase_client,
                "fetch_odds_rows_for_match_window",
            ) as mock_window,
        ):
            rows, scraped_at, event_id = odds_client.get_match_odds_rows(
                _EMPTY_SCORES,
                "Girona",
                "Villarreal",
            )

        mock_latest.assert_called_once_with(sport="soccer")
        mock_window.assert_not_called()
        assert len(rows) == 1
        assert rows[0]["home_team"] == "Girona"
        assert scraped_at == "2026-04-06T09:01:16+00:00"
        assert event_id == "ev_girona_villa"

    def test_no_match_date_misses_matches_in_older_scrapes(self) -> None:
        """Legacy path: matches in older scrapes are invisible (bug documented)."""
        with patch.object(
            odds_client.supabase_client,
            "fetch_latest_odds_rows",
            return_value={
                "scraped_at": "2026-04-06T09:01:16+00:00",
                "rows": [_J30_ROWS[3]],  # only Girona
            },
        ):
            rows, scraped_at, event_id = odds_client.get_match_odds_rows(
                _EMPTY_SCORES,
                "Mallorca",
                "Real Madrid",
            )
        # Legacy behavior: not found because the Mallorca row is in an
        # older scrape. This is THE BUG we fixed via the match_date path.
        assert rows == []
        assert event_id is None


class TestGetMatchOddsRowsWithMatchDate:
    """When match_date is provided, the window-based fetch is used."""

    def test_match_date_uses_window_fetch(self) -> None:
        """With match_date='2026-04-05', the window fetch is called."""
        with (
            patch.object(
                odds_client.supabase_client,
                "fetch_latest_odds_rows",
            ) as mock_latest,
            patch.object(
                odds_client.supabase_client,
                "fetch_odds_rows_for_match_window",
                return_value={
                    "window_start": "2026-04-01T00:00:00+00:00",
                    "window_end": "2026-04-06T23:59:59+00:00",
                    "latest_scraped_at": "2026-04-06T09:01:16+00:00",
                    "rows": _J30_ROWS,
                },
            ) as mock_window,
        ):
            rows, scraped_at, event_id = odds_client.get_match_odds_rows(
                _EMPTY_SCORES,
                "Mallorca",
                "Real Madrid",
                match_date="2026-04-05",
            )

        mock_latest.assert_not_called()
        mock_window.assert_called_once()
        call_kwargs = mock_window.call_args.kwargs
        assert mock_window.call_args.args[0] == "2026-04-05"
        assert call_kwargs["sport"] == "soccer"
        assert call_kwargs["days_before"] == 4
        assert call_kwargs["days_after"] == 1
        # Found the Mallorca row from the older scrape — THE FIX
        assert len(rows) == 1
        assert rows[0]["home_team"] == "Mallorca"
        assert event_id == "ev_mallorca_rm"

    def test_window_finds_older_scrape_matches(self) -> None:
        """The window-based fetch exposes matches scraped on earlier days.

        This is the J30 regression case: 'Vallecano vs Elche' is stored
        in odds_raw as 'Elche vs Rayo' (reversed home/away + short form
        'Rayo'). The combination of window fetch + TEAM_ALIASES resolves
        it when `scores` contains the canonical 'Vallecano' key.
        """
        with patch.object(
            odds_client.supabase_client,
            "fetch_odds_rows_for_match_window",
            return_value={
                "window_start": "2026-04-01T00:00:00+00:00",
                "window_end": "2026-04-06T23:59:59+00:00",
                "latest_scraped_at": "2026-04-06T09:01:16+00:00",
                "rows": _J30_ROWS,
            },
        ):
            rows, scraped_at, event_id = odds_client.get_match_odds_rows(
                _J30_SCORES,  # populated so fuzzy_name_search finds 'Vallecano'
                "Vallecano",
                "Elche",
                match_date="2026-04-05",
            )
        # Found the rows via reversed direction + TEAM_ALIASES lookup
        # ('rayo' → 'Vallecano' via fuzzy_name_search + TEAM_ALIASES)
        assert len(rows) == 2
        assert event_id == "ev_rayo_elche"

    def test_custom_window_days(self) -> None:
        """Window days_before / days_after are forwarded verbatim."""
        with patch.object(
            odds_client.supabase_client,
            "fetch_odds_rows_for_match_window",
            return_value={
                "window_start": "x",
                "window_end": "y",
                "latest_scraped_at": None,
                "rows": [],
            },
        ) as mock_window:
            odds_client.get_match_odds_rows(
                _EMPTY_SCORES,
                "Girona",
                "Villarreal",
                match_date="2026-04-06",
                window_days_before=7,
                window_days_after=2,
            )
        call_kwargs = mock_window.call_args.kwargs
        assert call_kwargs["days_before"] == 7
        assert call_kwargs["days_after"] == 2

    def test_empty_window_returns_empty(self) -> None:
        """No rows in window → empty result, no crash."""
        with patch.object(
            odds_client.supabase_client,
            "fetch_odds_rows_for_match_window",
            return_value={
                "window_start": "x",
                "window_end": "y",
                "latest_scraped_at": None,
                "rows": [],
            },
        ):
            rows, scraped_at, event_id = odds_client.get_match_odds_rows(
                _EMPTY_SCORES,
                "Girona",
                "Villarreal",
                match_date="2026-04-05",
            )
        assert rows == []
        assert scraped_at is None
        assert event_id is None

    def test_scraped_at_returned_is_latest_in_window(self) -> None:
        """The returned scraped_at is the latest_scraped_at from the window fetch."""
        with patch.object(
            odds_client.supabase_client,
            "fetch_odds_rows_for_match_window",
            return_value={
                "window_start": "2026-04-01T00:00:00+00:00",
                "window_end": "2026-04-06T23:59:59+00:00",
                "latest_scraped_at": "2026-04-06T09:01:16+00:00",
                "rows": _J30_ROWS,
            },
        ):
            _, scraped_at, _ = odds_client.get_match_odds_rows(
                _EMPTY_SCORES,
                "Mallorca",
                "Real Madrid",
                match_date="2026-04-05",
            )
        assert scraped_at == "2026-04-06T09:01:16+00:00"


class TestSupabaseClientWindowParsing:
    """Tests for fetch_odds_rows_for_match_window() date parsing.

    These tests use a mock Supabase client to avoid network dependencies.
    """

    def test_rejects_empty_match_date(self) -> None:
        """Empty match_date raises ValueError."""
        from selection import supabase_client

        with pytest.raises(ValueError, match="match_date is required"):
            supabase_client.fetch_odds_rows_for_match_window("")

    def test_rejects_invalid_format(self) -> None:
        """Non-ISO format raises ValueError."""
        from selection import supabase_client

        with pytest.raises(ValueError, match="must be ISO format"):
            supabase_client.fetch_odds_rows_for_match_window("05/04/2026")

    def test_accepts_iso_date(self) -> None:
        """Plain YYYY-MM-DD is accepted and produces a window."""
        from selection import supabase_client

        with patch.object(supabase_client, "get_client") as mock_client:
            mock_sb = mock_client.return_value

            class _FakeResp:
                data: list = []

            # Build a minimal chainable mock for sb.table(...).select(...).gte(...)....execute()
            mock_chain = mock_sb.table.return_value
            mock_chain.select.return_value = mock_chain
            mock_chain.eq.return_value = mock_chain
            mock_chain.gte.return_value = mock_chain
            mock_chain.lte.return_value = mock_chain
            mock_chain.range.return_value = mock_chain
            mock_chain.execute.return_value = _FakeResp()

            result = supabase_client.fetch_odds_rows_for_match_window("2026-04-05")

        assert "window_start" in result
        assert "window_end" in result
        assert result["rows"] == []
        # Window should span days_before=4 before to days_after=1 after
        assert result["window_start"].startswith("2026-04-01")
        assert result["window_end"].startswith("2026-04-06")

    def test_accepts_iso_datetime(self) -> None:
        """Full ISO timestamp with 'T' separator is accepted."""
        from selection import supabase_client

        with patch.object(supabase_client, "get_client") as mock_client:
            mock_sb = mock_client.return_value

            class _FakeResp:
                data: list = []

            mock_chain = mock_sb.table.return_value
            mock_chain.select.return_value = mock_chain
            mock_chain.eq.return_value = mock_chain
            mock_chain.gte.return_value = mock_chain
            mock_chain.lte.return_value = mock_chain
            mock_chain.range.return_value = mock_chain
            mock_chain.execute.return_value = _FakeResp()

            result = supabase_client.fetch_odds_rows_for_match_window(
                "2026-04-05T21:00:00+00:00"
            )

        assert result["window_start"].startswith("2026-04-01")
        assert result["window_end"].startswith("2026-04-06")
