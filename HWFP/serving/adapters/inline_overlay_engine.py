"""Adapter stub — InlineOverlayEngine (port: OverlayEngine)."""

from __future__ import annotations

from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.overlay import Overlay


class InlineOverlayEngine:
    """Stub for OverlayEngine. Raises NotImplementedError on all methods."""

    def compute(self, pmf: FoulPMF, odds: Odds) -> Overlay:
        raise NotImplementedError(
            "HWFP adapter stub — port=OverlayEngine; future change hwfp-inline-overlay-engine-adapter"
        )
