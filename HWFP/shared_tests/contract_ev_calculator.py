"""Contract tests for EVCalculator port (REQ-8, REQ-9)."""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds

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
_LINE = 22.5


@pytest.fixture(params=["fake", "real"], ids=["fake", "real"])
def ev_calculator(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_ev_calculator")
        return mod.FakeEVCalculator()
    mod = pytest.importorskip("HWFP.serving.adapters.kelly_ev_calculator")
    return mod.KellyEVCalculator()


def test_compute_returns_ev_result(ev_calculator):
    result = ev_calculator.compute(_PMF, _ODDS, _LINE)
    assert isinstance(result, EVResult)


def test_compute_ev_is_finite(ev_calculator):
    result = ev_calculator.compute(_PMF, _ODDS, _LINE)
    assert math.isfinite(result.ev)


def test_compute_fair_prob_in_unit_interval(ev_calculator):
    result = ev_calculator.compute(_PMF, _ODDS, _LINE)
    assert 0.0 <= result.fair_prob <= 1.0


def test_compute_book_prob_in_open_interval(ev_calculator):
    result = ev_calculator.compute(_PMF, _ODDS, _LINE)
    assert 0.0 < result.book_prob < 1.0
