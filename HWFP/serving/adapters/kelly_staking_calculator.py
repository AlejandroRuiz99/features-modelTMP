"""Kelly Staking Calculator adapter — implements StakingCalculator port."""

from __future__ import annotations

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.stake_result import StakeResult

_KELLY_CAP: float = 0.30
_MIN_STAKE_EUROS: float = 5.0


class KellyStakingCalculator:
    """Compute recommended stake using the full Kelly criterion.

    kelly_fraction = ev × book_prob / (1 − book_prob)
      (equivalent to ev / (decimal − 1), derived from book_prob = 1/decimal).
    Fraction is capped at 30% of bankroll.
    Bets below €5 are suppressed to zero.
    """

    def compute(self, ev: EVResult, bankroll: float) -> StakeResult:
        """Compute StakeResult for a given EVResult and available bankroll."""
        if ev.ev <= 0.0:
            return StakeResult(
                match_id=ev.match_id,
                market=ev.market,
                stake=0.0,
                kelly_fraction=0.0,
                bankroll_used=bankroll,
            )

        kelly_fraction = ev.ev * ev.book_prob / (1.0 - ev.book_prob)
        kelly_fraction = min(kelly_fraction, _KELLY_CAP)
        stake = kelly_fraction * bankroll

        if stake < _MIN_STAKE_EUROS:
            return StakeResult(
                match_id=ev.match_id,
                market=ev.market,
                stake=0.0,
                kelly_fraction=0.0,
                bankroll_used=bankroll,
            )

        return StakeResult(
            match_id=ev.match_id,
            market=ev.market,
            stake=stake,
            kelly_fraction=kelly_fraction,
            bankroll_used=bankroll,
        )
