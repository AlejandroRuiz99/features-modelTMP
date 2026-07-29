"""FakeFoulModel — returns a fixed deterministic PMF. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.foul_pmf import FoulPMF

_PMF: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05)
_BIN_EDGES: tuple[int, ...] = (0, 15, 20, 22, 24, 26, 30, 40)
_FIXED_PMF = FoulPMF(pmf=_PMF, bin_edges=_BIN_EDGES)


class FakeFoulModel:
    """Returns a fixed PMF regardless of input features.

    PMF: (0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05) over bins (0,15,20,22,24,26,30,40).
    Sum == 1.0 exactly. Deterministic. Zero I/O.
    """

    def predict(self, features: tuple[float, ...]) -> FoulPMF:
        return _FIXED_PMF
