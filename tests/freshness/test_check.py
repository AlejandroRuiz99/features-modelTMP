"""
tests/freshness/test_check.py — TDD for freshness.check

Tests (T4.1):
  1. Gap ≤ 14 days → FreshnessResult.status="ok".
  2. Gap > 14 days → FreshnessResult.status="warning".
  3. Gap=0 (latest match = earliest match date) → status="ok".
  4. No fecha at all (empty table) → status="error".
  5. FreshnessResult has expected fields: status, last_date, gap_days.
  6. Mocked Supabase responses — no live DB.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from freshness.check import FreshnessResult, check_freshness


def _make_supabase_mock(last_date: str | None) -> MagicMock:
    """Create a mock Supabase client that returns a given MAX fecha."""
    client = MagicMock()

    # Simulate: client.table("matches").select("match_date").order("match_date", desc=True).limit(1).execute()
    execute_result = MagicMock()
    if last_date is not None:
        execute_result.data = [{"match_date": last_date}]
    else:
        execute_result.data = []

    q = MagicMock()
    q.execute.return_value = execute_result
    q.select.return_value = q
    q.order.return_value = q
    q.limit.return_value = q

    client.table.return_value = q
    return client


class TestFreshnessResultFields:
    def test_result_has_status_field(self) -> None:
        """FreshnessResult has a status field."""
        r = FreshnessResult(status="ok", last_date="2026-05-01", gap_days=3)
        assert r.status == "ok"

    def test_result_has_last_date_field(self) -> None:
        """FreshnessResult has a last_date field."""
        r = FreshnessResult(status="ok", last_date="2026-05-01", gap_days=3)
        assert r.last_date == "2026-05-01"

    def test_result_has_gap_days_field(self) -> None:
        """FreshnessResult has a gap_days field."""
        r = FreshnessResult(status="ok", last_date="2026-05-01", gap_days=3)
        assert r.gap_days == 3


class TestCheckFreshness:
    def test_gap_zero_returns_ok(self) -> None:
        """Gap=0 (stats current with match date) → status=ok."""
        match_date = date(2026, 5, 4)
        last_stat_date = str(match_date)
        client = _make_supabase_mock(last_stat_date)

        result = check_freshness(client, earliest_match_date=match_date)

        assert result.status == "ok"
        assert result.gap_days == 0

    def test_gap_6_days_returns_ok(self) -> None:
        """Gap of 6 days → status=ok (≤ 14 day threshold)."""
        match_date = date(2026, 5, 10)
        last_stat_date = str(match_date - timedelta(days=6))
        client = _make_supabase_mock(last_stat_date)

        result = check_freshness(client, earliest_match_date=match_date)

        assert result.status == "ok"
        assert result.gap_days == 6

    def test_gap_14_days_returns_ok(self) -> None:
        """Gap of exactly 14 days → status=ok (boundary is ≤ 14)."""
        match_date = date(2026, 5, 20)
        last_stat_date = str(match_date - timedelta(days=14))
        client = _make_supabase_mock(last_stat_date)

        result = check_freshness(client, earliest_match_date=match_date)

        assert result.status == "ok"
        assert result.gap_days == 14

    def test_gap_15_days_returns_warning(self) -> None:
        """Gap of 15 days → status=warning (> 14 threshold)."""
        match_date = date(2026, 5, 20)
        last_stat_date = str(match_date - timedelta(days=15))
        client = _make_supabase_mock(last_stat_date)

        result = check_freshness(client, earliest_match_date=match_date)

        assert result.status == "warning"
        assert result.gap_days == 15

    def test_gap_20_days_returns_warning(self) -> None:
        """Gap of 20 days → status=warning."""
        match_date = date(2026, 5, 25)
        last_stat_date = str(match_date - timedelta(days=20))
        client = _make_supabase_mock(last_stat_date)

        result = check_freshness(client, earliest_match_date=match_date)

        assert result.status == "warning"
        assert result.gap_days == 20

    def test_no_data_returns_error(self) -> None:
        """Empty tabla (no match_date rows) → status=error."""
        client = _make_supabase_mock(last_date=None)

        result = check_freshness(client, earliest_match_date=date(2026, 5, 10))

        assert result.status == "error"
        assert result.last_date is None

    def test_last_date_preserved_in_result(self) -> None:
        """last_date in result matches the DB value."""
        match_date = date(2026, 5, 10)
        last_stat_date = "2026-05-01"
        client = _make_supabase_mock(last_stat_date)

        result = check_freshness(client, earliest_match_date=match_date)

        assert result.last_date == "2026-05-01"

    def test_supabase_exception_returns_error(self) -> None:
        """If Supabase raises an exception → FreshnessResult.status=error."""
        client = MagicMock()
        client.table.side_effect = RuntimeError("Supabase connection failed")

        result = check_freshness(client, earliest_match_date=date(2026, 5, 10))

        assert result.status == "error"
