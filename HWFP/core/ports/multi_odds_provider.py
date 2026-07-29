"""MultiOddsProvider port — aggregate odds from multiple bookmakers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from HWFP.core.domain.odds import Odds


@runtime_checkable
class MultiOddsProvider(Protocol):
    """Provides aggregated odds from multiple bookmaker providers.

    AggregatedOddsProvider in serving/adapters satisfies this protocol via duck typing.
    """

    def get_odds(self, match_id: str, market: str) -> list[Odds]: ...
    def best_odds(self, match_id: str, market: str) -> Odds | None: ...
