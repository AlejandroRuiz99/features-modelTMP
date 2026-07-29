"""OddsProvider port — fetch bookmaker odds."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.odds import Odds


class OddsProvider(Protocol):
    """Fetch bookmaker odds for a match/market pair.

    Raises:
        OddsNotFoundError: If no odds exist for the given match/market pair.
    Invariant:
        Returned Odds.decimal > 1.0.
    """

    def get_odds(self, match_id: str, market: str) -> Odds: ...
