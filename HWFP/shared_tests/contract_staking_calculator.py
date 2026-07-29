"""Contract tests for StakingCalculator port (REQ-8, REQ-9)."""

from __future__ import annotations

import pytest

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.stake_result import StakeResult

_EV = EVResult(
    match_id="M1",
    market="fouls_over_under",
    line=22.5,
    side="over",
    fair_prob=0.60,
    book_prob=0.5128,
    ev=0.17,
)
_BANKROLL = 1000.0


@pytest.fixture(params=["fake", "real"], ids=["fake", "real"])
def staking_calculator(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_staking_calculator")
        return mod.FakeStakingCalculator()
    mod = pytest.importorskip("HWFP.serving.adapters.kelly_staking_calculator")
    return mod.KellyStakingCalculator()


def test_compute_returns_stake_result(staking_calculator):
    result = staking_calculator.compute(_EV, _BANKROLL)
    assert isinstance(result, StakeResult)


def test_compute_stake_in_bankroll_range(staking_calculator):
    result = staking_calculator.compute(_EV, _BANKROLL)
    assert 0.0 <= result.stake <= _BANKROLL


def test_compute_kelly_fraction_in_unit_interval(staking_calculator):
    result = staking_calculator.compute(_EV, _BANKROLL)
    assert 0.0 <= result.kelly_fraction <= 1.0
