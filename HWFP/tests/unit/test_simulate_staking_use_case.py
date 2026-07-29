"""Unit tests for SimulateStakingUseCase — B7b TDD RED gate (T7b.1)."""

from __future__ import annotations

from HWFP.core.application.simulate_staking import SimulateInput, SimulateStakingUseCase
from HWFP.core.domain.ev_result import EVResult
from HWFP.serving.fakes.fake_staking_calculator import FakeStakingCalculator


def _make_ev(match_id: str = "M1", ev_value: float = 0.17) -> EVResult:
    return EVResult(
        match_id=match_id,
        market="fouls_over_under",
        line=22.5,
        side="over",
        fair_prob=0.6,
        book_prob=0.52,
        ev=ev_value,
    )


def test_rolling_bankroll_updates() -> None:
    """T7b.1: Rolling bankroll changes per Kelly formula across fixed EVResult sequence."""
    ev = _make_ev(ev_value=0.17)
    inp = SimulateInput(ev_sequence=(ev, ev), initial_bankroll=1000.0)
    uc = SimulateStakingUseCase(staking=FakeStakingCalculator(kelly_fraction=0.25))
    out = uc.execute(inp)

    # Positive EV → final bankroll grows
    assert out.final_bankroll > 1000.0
    # Two EVResults → two stakes produced
    assert len(out.stakes) == 2
    # Rolling: second stake used the updated (larger) bankroll
    assert out.stakes[1].bankroll_used > out.stakes[0].bankroll_used


def test_negative_ev_reduces_bankroll() -> None:
    ev = _make_ev(ev_value=-0.10)
    inp = SimulateInput(ev_sequence=(ev, ev), initial_bankroll=1000.0)
    uc = SimulateStakingUseCase(staking=FakeStakingCalculator(kelly_fraction=0.25))
    out = uc.execute(inp)

    assert out.final_bankroll < 1000.0


def test_empty_sequence_preserves_bankroll() -> None:
    inp = SimulateInput(ev_sequence=(), initial_bankroll=1000.0)
    uc = SimulateStakingUseCase(staking=FakeStakingCalculator())
    out = uc.execute(inp)

    assert out.final_bankroll == 1000.0
    assert out.stakes == ()
