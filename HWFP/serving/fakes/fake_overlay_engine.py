"""FakeOverlayEngine — pure arithmetic overlay. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.overlay import Overlay


class FakeOverlayEngine:
    """Computes overlay via pure arithmetic: implied = 1/decimal, fair from PMF tail.

    For 'over': sums bins whose upper edge exceeds the line.
    For 'under': sums bins whose lower edge is below the line.
    """

    def compute(self, pmf: FoulPMF, odds: Odds) -> Overlay:
        implied_prob = 1.0 / odds.decimal
        if odds.side == "over":
            fair_prob = sum(
                p for i, p in enumerate(pmf.pmf) if pmf.bin_edges[i + 1] > odds.line
            )
        else:
            fair_prob = sum(
                p for i, p in enumerate(pmf.pmf) if pmf.bin_edges[i] < odds.line
            )
        return Overlay(
            match_id=odds.match_id,
            market=odds.market,
            fair_prob=fair_prob,
            implied_prob=implied_prob,
            edge=fair_prob - implied_prob,
        )
