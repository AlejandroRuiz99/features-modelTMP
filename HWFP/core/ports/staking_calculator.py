"""StakingCalculator port — compute stake size from EV and bankroll."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.stake_result import StakeResult


class StakingCalculator(Protocol):
    """Compute recommended stake from an EVResult and available bankroll.

    Invariant:
        0 <= stake <= bankroll.
    """

    def compute(self, ev: EVResult, bankroll: float) -> StakeResult: ...
