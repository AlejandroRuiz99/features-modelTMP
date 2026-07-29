"""FakeStakingCalculator — fractional Kelly with fixed fraction. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.stake_result import StakeResult

_DEFAULT_KELLY_FRACTION: float = 0.25


class FakeStakingCalculator:
    """Fractional Kelly stake: stake = kelly_fraction × bankroll.

    Invariant: 0 <= stake <= bankroll.
    Happy-path fraction for golden e2e: 0.25.
    """

    def __init__(self, kelly_fraction: float = _DEFAULT_KELLY_FRACTION) -> None:
        self._kelly_fraction = kelly_fraction

    def compute(self, ev: EVResult, bankroll: float) -> StakeResult:
        stake = min(max(self._kelly_fraction * bankroll, 0.0), bankroll)
        return StakeResult(
            match_id=ev.match_id,
            market=ev.market,
            stake=stake,
            kelly_fraction=self._kelly_fraction,
            bankroll_used=bankroll,
        )
