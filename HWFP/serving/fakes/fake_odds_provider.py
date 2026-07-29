"""FakeOddsProvider — pre-loaded in-memory odds. Zero I/O."""

from __future__ import annotations

from datetime import datetime

from HWFP.core.domain.exceptions import OddsNotFoundError
from HWFP.core.domain.odds import Odds

_FETCHED_AT = datetime(2026, 6, 16, 18, 0, 0)


class FakeOddsProvider:
    """Looks up odds from an in-memory dict keyed by (match_id, market).

    Happy-path fixture: (M1, fouls_over_under) → line=22.5, side=over, decimal=1.95.
    Raises OddsNotFoundError for any unknown (match_id, market) pair.
    """

    def __init__(self, odds_map: dict[tuple[str, str], Odds] | None = None) -> None:
        self._odds: dict[tuple[str, str], Odds] = (
            odds_map if odds_map is not None else {}
        )

    @classmethod
    def with_fixture(cls) -> FakeOddsProvider:
        """Return instance pre-loaded with golden e2e odds for M1 fouls_over_under."""
        return cls(
            odds_map={
                ("M1", "fouls_over_under"): Odds(
                    match_id="M1",
                    market="fouls_over_under",
                    line=22.5,
                    side="over",
                    decimal=1.95,
                    bookmaker="codere",
                    fetched_at=_FETCHED_AT,
                )
            }
        )

    def get_odds(self, match_id: str, market: str) -> Odds:
        try:
            return self._odds[(match_id, market)]
        except KeyError:
            raise OddsNotFoundError(
                f"Odds not found for match={match_id!r}, market={market!r}"
            )
