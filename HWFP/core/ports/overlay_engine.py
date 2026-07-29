"""OverlayEngine port — compute model-vs-market overlay."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.overlay import Overlay


class OverlayEngine(Protocol):
    """Compute model-vs-market overlay from a PMF and odds.

    Pure function — no I/O, no side effects.
    """

    def compute(self, pmf: FoulPMF, odds: Odds) -> Overlay: ...
