"""Contract tests for OverlayEngine port (REQ-8, REQ-9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.overlay import Overlay

_PMF = FoulPMF(
    pmf=(0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05),
    bin_edges=(0, 15, 20, 22, 24, 26, 30, 40),
)
_ODDS = Odds(
    match_id="M1",
    market="fouls_over_under",
    line=22.5,
    side="over",
    decimal=1.95,
    bookmaker="codere",
    fetched_at=datetime(2026, 6, 16, 20, 0, 0),
)


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def overlay_engine(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_overlay_engine")
        return mod.FakeOverlayEngine()
    pytest.importorskip("HWFP.serving.adapters.inline_overlay_engine")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_compute_returns_overlay(overlay_engine):
    result = overlay_engine.compute(_PMF, _ODDS)
    assert isinstance(result, Overlay)


def test_compute_is_pure(overlay_engine):
    assert overlay_engine.compute(_PMF, _ODDS) == overlay_engine.compute(_PMF, _ODDS)


def test_overlay_engine_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.inline_overlay_engine")
    adapter = mod.InlineOverlayEngine()
    with pytest.raises(NotImplementedError):
        adapter.compute(_PMF, _ODDS)
