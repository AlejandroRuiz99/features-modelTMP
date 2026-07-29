"""Aggregates odds from multiple OddsProvider instances."""

from __future__ import annotations

from HWFP.core.domain.odds import Odds
from HWFP.core.ports.odds_provider import OddsProvider


class AggregatedOddsProvider:
    """Collects odds from multiple providers; silently skips stubs (NotImplementedError).

    Note: get_odds() returns list[Odds], not a single Odds, because the aggregator
    is a collector — not a direct OddsProvider implementor. Use best_odds() to
    get the single highest-decimal result.
    """

    def __init__(self, providers: list[OddsProvider]) -> None:
        self._providers = providers

    def get_odds(self, match_id: str, market: str) -> list[Odds]:
        """Return odds from all non-stub providers. Empty list if all are stubs."""
        all_odds: list[Odds] = []
        for provider in self._providers:
            try:
                all_odds.append(provider.get_odds(match_id, market))
            except NotImplementedError:
                continue
        return all_odds

    def best_odds(self, match_id: str, market: str) -> Odds | None:
        """Return the Odds with the highest decimal, or None if all providers are stubs."""
        all_odds = self.get_odds(match_id, market)
        if not all_odds:
            return None
        return max(all_odds, key=lambda o: o.decimal)
