"""EVCalculator port — compute expected value for a bet."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds


class EVCalculator(Protocol):
    """Compute expected value for a bet given a PMF, odds, and line.

    Pure function — no I/O, no side effects.
    """

    def compute(self, pmf: FoulPMF, odds: Odds, line: float) -> EVResult: ...
