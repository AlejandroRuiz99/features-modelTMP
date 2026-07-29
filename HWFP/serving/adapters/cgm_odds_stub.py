"""CasinoGranMadrid OddsProvider — mock data pending real XHR scraper."""

from __future__ import annotations

from datetime import datetime, timezone

from HWFP.core.domain.odds import Odds

# To implement: inspect https://www.casinogranmadrid.es DevTools → Network → XHR
_MOCK_DECIMAL = 1.87
_MOCK_LINE = 9.5
_MOCK_SIDE = "over"


class CGMOddsStub:
    """Mock OddsProvider for CasinoGranMadrid. Returns deterministic odds for pipeline testing."""

    def get_odds(self, match_id: str, market: str) -> Odds:
        return Odds(
            match_id=match_id,
            market=market,
            line=_MOCK_LINE,
            side=_MOCK_SIDE,
            decimal=_MOCK_DECIMAL,
            bookmaker="cgm",
            fetched_at=datetime.now(timezone.utc),
        )
