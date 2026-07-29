"""Supabase adapter — StateProvider port.

Fetches match and team state from the Supabase `matches` table.
Connection is lazy: no network call is made until the first query.
Queries run in a thread-pool executor so a per-call timeout can be enforced.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from HWFP.core.domain.exceptions import StateNotFoundError
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState

logger = logging.getLogger(__name__)

_MATCH_COLS = (
    "match_id,match_date,season,referee,"
    "home_team,away_team,"
    "fouls_home,fouls_away"
)
_FORM_WINDOW = 5


class SupabaseStateAdapter:
    """StateProvider backed by the Supabase `matches` table.

    Args:
        url: Supabase project URL (SUPABASE_URL).
        key: Supabase service-role or anon key (SUPABASE_KEY).
        timeout_seconds: Per-call network timeout in seconds.
        _client: Injected client for testing; skips lazy init and executor.
    """

    def __init__(
        self,
        url: str,
        key: str,
        timeout_seconds: float = 10.0,
        *,
        _client: object | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._timeout = timeout_seconds
        self._injected: object | None = _client
        self._client: Any = _client
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            None
            if _client is not None
            else concurrent.futures.ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="supa-state"
            )
        )

    def _get_client(self) -> Any:
        if self._client is None:
            from supabase import create_client

            self._client = create_client(self._url, self._key)
        return self._client

    def _run(self, fn: Callable[[], Any]) -> Any:
        """Run fn() synchronously (test) or in the executor with a timeout (prod)."""
        if self._executor is None:
            return fn()
        future = self._executor.submit(fn)
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"Supabase call timed out after {self._timeout}s"
            ) from exc

    def get_match(self, match_id: str) -> Match:
        """Fetch a single match by ID.

        Args:
            match_id: Primary key in the `matches` table.

        Returns:
            Match domain object.

        Raises:
            StateNotFoundError: If no row with this match_id exists.
        """
        sb = self._get_client()

        def _query() -> Any:
            return (
                sb.table("matches")
                .select(_MATCH_COLS)
                .eq("match_id", match_id)
                .limit(1)
                .execute()
            )

        response = self._run(_query)
        if not response.data:
            raise StateNotFoundError(f"Match '{match_id}' not found in Supabase")
        return _row_to_match(response.data[0])

    def get_team_state(self, team_id: str, as_of: datetime) -> TeamState:
        """Derive TeamState from the team's recent match history.

        Fetches up to `_FORM_WINDOW` matches played before `as_of` where the
        team appeared as home or away, then computes rolling foul averages.

        Args:
            team_id: Team name as stored in `home_team` / `away_team` columns.
            as_of: Upper bound — only matches with match_date < as_of are used.

        Returns:
            TeamState with rolling averages over the most recent form window.
        """
        sb = self._get_client()
        as_of_str = as_of.isoformat()
        cols = "match_id,match_date,home_team,away_team,fouls_home,fouls_away"

        def _home_query() -> Any:
            return (
                sb.table("matches")
                .select(cols)
                .eq("home_team", team_id)
                .lt("match_date", as_of_str)
                .order("match_date", desc=True)
                .limit(_FORM_WINDOW)
                .execute()
            )

        def _away_query() -> Any:
            return (
                sb.table("matches")
                .select(cols)
                .eq("away_team", team_id)
                .lt("match_date", as_of_str)
                .order("match_date", desc=True)
                .limit(_FORM_WINDOW)
                .execute()
            )

        home_resp = self._run(_home_query)
        away_resp = self._run(_away_query)

        rows: list[dict[str, Any]] = (home_resp.data or []) + (away_resp.data or [])
        rows.sort(key=lambda r: r.get("match_date") or "", reverse=True)
        rows = rows[:_FORM_WINDOW]

        return _rows_to_team_state(team_id, as_of, rows)


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------


def _row_to_match(row: dict[str, Any]) -> Match:
    date_str = row.get("match_date") or ""
    try:
        kickoff = datetime.fromisoformat(date_str)
    except ValueError:
        kickoff = datetime(1970, 1, 1)

    return Match(
        match_id=row["match_id"],
        home_team_id=row["home_team"],
        away_team_id=row["away_team"],
        kickoff=kickoff,
        referee_id=row.get("referee") or "unknown",
        competition_id=row.get("season") or "unknown",
    )


def _rows_to_team_state(
    team_id: str,
    as_of: datetime,
    rows: list[dict[str, Any]],
) -> TeamState:
    fouls: list[float] = []
    conceded: list[float] = []
    for row in rows:
        if row.get("home_team") == team_id:
            fouls.append(float(row.get("fouls_home") or 0))
            conceded.append(float(row.get("fouls_away") or 0))
        else:
            fouls.append(float(row.get("fouls_away") or 0))
            conceded.append(float(row.get("fouls_home") or 0))
    n = len(fouls)
    return TeamState(
        team_id=team_id,
        as_of=as_of,
        avg_fouls_per_match=sum(fouls) / n if n else 0.0,
        avg_fouls_conceded=sum(conceded) / n if n else 0.0,
        form_window=n,
    )
