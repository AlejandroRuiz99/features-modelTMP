"""
freshness.check — Stats freshness check (Step 1 of predecir-jornada-v2).

Queries Supabase for the most recent match date in the database and compares
to the earliest match date the user wants to predict. Gap > 14 days → warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = ["FreshnessResult", "check_freshness"]

_FRESHNESS_THRESHOLD_DAYS = 14


@dataclass
class FreshnessResult:
    """Result of the stats freshness check.

    Fields:
        status: "ok" | "warning" | "error"
            ok      → gap ≤ 14 days
            warning → gap > 14 days
            error   → no data found or Supabase unreachable
        last_date: ISO date string of the most recent match in DB, or None.
        gap_days: Number of days between last_date and earliest_match_date.
    """

    status: str
    last_date: str | None
    gap_days: int


def check_freshness(
    supabase_client: Any,
    earliest_match_date: date,
) -> FreshnessResult:
    """Check if stats are fresh enough for the given match dates.

    Queries Supabase for the most recent match date (MAX(match_date)).
    Compares to earliest_match_date to compute the gap.

    Args:
        supabase_client: Supabase client instance (or mock).
        earliest_match_date: The earliest date of matches in this prediction run.

    Returns:
        FreshnessResult with status, last_date, and gap_days.
    """
    try:
        result = (
            supabase_client.table("matches")
            .select("match_date")
            .order("match_date", desc=True)
            .limit(1)
            .execute()
        )
        rows = result.data if result.data else []
    except Exception:
        return FreshnessResult(status="error", last_date=None, gap_days=0)

    if not rows:
        return FreshnessResult(status="error", last_date=None, gap_days=0)

    last_date_str = rows[0].get("match_date")
    if not last_date_str:
        return FreshnessResult(status="error", last_date=None, gap_days=0)

    try:
        last_date = date.fromisoformat(str(last_date_str)[:10])
    except ValueError:
        return FreshnessResult(status="error", last_date=str(last_date_str), gap_days=0)

    gap_days = (earliest_match_date - last_date).days
    # Negative gap (stats are MORE recent than match date) → gap is 0
    gap_days = max(0, gap_days)

    status = "ok" if gap_days <= _FRESHNESS_THRESHOLD_DAYS else "warning"

    return FreshnessResult(
        status=status,
        last_date=str(last_date_str)[:10],
        gap_days=gap_days,
    )
