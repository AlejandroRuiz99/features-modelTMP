"""Kelly EV Calculator adapter — implements EVCalculator port."""

from __future__ import annotations

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds


class KellyEVCalculator:
    """Compute expected value using the Kelly edge formula.

    fair_prob:
      'over'  → sum of PMF bins whose upper edge strictly exceeds the line.
      'under' → sum of PMF bins whose lower edge is strictly below the line.
    book_prob = 1 / decimal.
    ev = decimal × fair_prob − 1.
    """

    def compute(self, pmf: FoulPMF, odds: Odds, line: float) -> EVResult:
        """Compute EVResult for a given PMF, odds quote, and line."""
        book_prob = 1.0 / odds.decimal
        if odds.side == "over":
            fair_prob = sum(
                p for i, p in enumerate(pmf.pmf) if pmf.bin_edges[i + 1] > line
            )
        else:
            fair_prob = sum(
                p for i, p in enumerate(pmf.pmf) if pmf.bin_edges[i] < line
            )
        return EVResult(
            match_id=odds.match_id,
            market=odds.market,
            line=line,
            side=odds.side,
            fair_prob=fair_prob,
            book_prob=book_prob,
            ev=odds.decimal * fair_prob - 1.0,
        )
