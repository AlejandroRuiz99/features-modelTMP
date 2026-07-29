"""FakeEVCalculator — ev = decimal × fair_prob − 1. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds


class FakeEVCalculator:
    """Pure EV formula: ev = decimal × fair_prob − 1.

    fair_prob derived from PMF tail:
      'over'  → sum bins whose upper edge exceeds the line.
      'under' → sum bins whose lower edge is below the line.
    book_prob = 1 / decimal.
    """

    def compute(self, pmf: FoulPMF, odds: Odds, line: float) -> EVResult:
        book_prob = 1.0 / odds.decimal
        if odds.side == "over":
            fair_prob = sum(
                p for i, p in enumerate(pmf.pmf) if pmf.bin_edges[i + 1] > line
            )
        else:
            fair_prob = sum(p for i, p in enumerate(pmf.pmf) if pmf.bin_edges[i] < line)
        return EVResult(
            match_id=odds.match_id,
            market=odds.market,
            line=line,
            side=odds.side,
            fair_prob=fair_prob,
            book_prob=book_prob,
            ev=odds.decimal * fair_prob - 1.0,
        )
