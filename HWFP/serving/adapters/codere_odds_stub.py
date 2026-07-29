"""Codere OddsProvider — mock data pending real XHR scraper."""

from __future__ import annotations

from datetime import datetime, timezone

from HWFP.core.domain.odds import Odds

# To implement the real scraper: inspect https://www.codere.es DevTools → Network → XHR
# to find the JSON endpoint that returns live odds, then replace _MOCK_* with live fetches.
_MOCK_DECIMAL = 1.90
_MOCK_LINE = 9.5
_MOCK_SIDE = "over"


class CodereOddsStub:
    """Mock OddsProvider for Codere. Returns deterministic odds so the full pipeline runs.

    Replace get_odds() body with real XHR call when the endpoint is identified.
    """

    def get_odds(self, match_id: str, market: str) -> Odds:
        return Odds(
            match_id=match_id,
            market=market,
            line=_MOCK_LINE,
            side=_MOCK_SIDE,
            decimal=_MOCK_DECIMAL,
            bookmaker="codere",
            fetched_at=datetime.now(timezone.utc),
        )
