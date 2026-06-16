"""
Unit tests for supabase_client.fetch_matches_for_season (T2.1).

Uses mocked supabase client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestFetchMatchesForSeason:
    """Tests for the new fetch_matches_for_season function."""

    def test_returns_list_of_dicts(self) -> None:
        """fetch_matches_for_season returns list of dicts."""
        mock_rows = [
            {
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "match_date": "2025-08-17",
            },
            {
                "home_team": "Ath Madrid",
                "away_team": "Getafe",
                "match_date": "2025-08-17",
            },
        ]

        with patch("selection.supabase_client.get_client") as mock_get_client:
            mock_sb = MagicMock()
            mock_get_client.return_value = mock_sb

            # Mock paginate_select behavior
            mock_sb.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = mock_rows

            # Import here to avoid module-level supabase import
            from selection.supabase_client import fetch_matches_for_season

            with patch(
                "selection.supabase_client._paginate_select",
                return_value=mock_rows,
            ):
                result = fetch_matches_for_season("2025-26")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_passes_season_filter(self) -> None:
        """fetch_matches_for_season passes season_str as filter."""
        with (
            patch("selection.supabase_client._paginate_select") as mock_paginate,
            patch("selection.supabase_client.get_client"),
        ):
            mock_paginate.return_value = []
            from selection.supabase_client import fetch_matches_for_season

            fetch_matches_for_season("2024-25")

            # Verify _paginate_select was called with the matches table
            call_args = mock_paginate.call_args
            assert call_args is not None
            # First positional arg should be the supabase client, second is "matches"
            assert "matches" in call_args.args

    def test_empty_season_returns_empty_list(self) -> None:
        """Returns empty list when no matches found for season."""
        with (
            patch(
                "selection.supabase_client._paginate_select",
                return_value=[],
            ),
            patch("selection.supabase_client.get_client"),
        ):
            from selection.supabase_client import fetch_matches_for_season

            result = fetch_matches_for_season("2020-21")
            assert result == []
