"""1xBet OddsProvider — mock data pending real XHR scraper."""

from __future__ import annotations

from datetime import datetime, timezone

from HWFP.core.domain.odds import Odds

# To implement: inspect https://1xbet.es DevTools → Network → XHR
_MOCK_DECIMAL = 1.92
_MOCK_LINE = 9.5
_MOCK_SIDE = "over"


class OnexBetOddsStub:
    """Mock OddsProvider for 1xBet. Returns deterministic odds for pipeline testing.

    1xbet typically offers slightly better decimals than Codere/CGM,
    so this stub wins best_odds() selection by default.
    """

    def get_odds(self, match_id: str, market: str) -> Odds:
        return Odds(
            match_id=match_id,
            market=market,
            line=_MOCK_LINE,
            side=_MOCK_SIDE,
            decimal=_MOCK_DECIMAL,
            bookmaker="1xbet",
            fetched_at=datetime.now(timezone.utc),
        )
