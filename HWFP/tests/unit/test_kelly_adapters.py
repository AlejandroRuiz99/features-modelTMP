"""Unit tests for KellyEVCalculator and KellyStakingCalculator adapters."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.stake_result import StakeResult
from HWFP.serving.adapters.kelly_ev_calculator import KellyEVCalculator
from HWFP.serving.adapters.kelly_staking_calculator import KellyStakingCalculator

# ── shared fixtures ──────────────────────────────────────────────────────────

_PMF = FoulPMF(
    pmf=(0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05),
    bin_edges=(0, 15, 20, 22, 24, 26, 30, 40),
)
# bins layout:
#  i=0 [0,15)   i=1 [15,20)  i=2 [20,22)  i=3 [22,24)
#  i=4 [24,26)  i=5 [26,30)  i=6 [30,40)
# For line=22.5:
#   over  → upper_edge > 22.5 → bins 3,4,5,6 → 0.30+0.20+0.10+0.05 = 0.65
#   under → lower_edge < 22.5 → bins 0,1,2,3 → 0.05+0.10+0.20+0.30 = 0.65


def _make_odds(*, side: str = "over", decimal: float = 1.95) -> Odds:
    return Odds(
        match_id="M1",
        market="fouls_over_under",
        line=22.5,
        side=side,
        decimal=decimal,
        bookmaker="codere",
        fetched_at=datetime(2026, 6, 16, 20, 0, 0),
    )


def _make_ev(
    *,
    ev: float = 0.17,
    fair_prob: float = 0.60,
    book_prob: float = 0.5128,
) -> EVResult:
    return EVResult(
        match_id="M1",
        market="fouls_over_under",
        line=22.5,
        side="over",
        fair_prob=fair_prob,
        book_prob=book_prob,
        ev=ev,
    )


# ── KellyEVCalculator ────────────────────────────────────────────────────────


class TestKellyEVCalculator:
    def setup_method(self) -> None:
        self.calc = KellyEVCalculator()

    def test_returns_ev_result_instance(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(), 22.5)
        assert isinstance(result, EVResult)

    def test_metadata_fields_come_from_odds(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(), 22.5)
        assert result.match_id == "M1"
        assert result.market == "fouls_over_under"
        assert result.line == 22.5
        assert result.side == "over"

    def test_fair_prob_over_sums_bins_whose_upper_edge_exceeds_line(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(side="over"), 22.5)
        assert abs(result.fair_prob - 0.65) < 1e-9

    def test_fair_prob_under_sums_bins_whose_lower_edge_is_below_line(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(side="under"), 22.5)
        assert abs(result.fair_prob - 0.65) < 1e-9

    def test_book_prob_is_inverse_of_decimal(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(decimal=2.0), 22.5)
        assert abs(result.book_prob - 0.5) < 1e-9

    def test_ev_equals_decimal_times_fair_prob_minus_one(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(side="over", decimal=2.0), 22.5)
        expected = 2.0 * result.fair_prob - 1.0
        assert abs(result.ev - expected) < 1e-9

    def test_ev_positive_when_model_has_edge(self) -> None:
        # fair_prob=0.65, decimal=1.95 → ev = 1.95*0.65-1 = 0.2675 > 0
        result = self.calc.compute(_PMF, _make_odds(side="over", decimal=1.95), 22.5)
        assert result.ev > 0.0

    def test_ev_negative_when_model_has_no_edge(self) -> None:
        # fair_prob_under=0.65, decimal=1.30 → ev = 1.30*0.65-1 = -0.155 < 0
        result = self.calc.compute(_PMF, _make_odds(side="under", decimal=1.30), 22.5)
        assert result.ev < 0.0

    def test_fair_prob_is_in_unit_interval(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(), 22.5)
        assert 0.0 <= result.fair_prob <= 1.0

    def test_book_prob_is_in_open_unit_interval(self) -> None:
        result = self.calc.compute(_PMF, _make_odds(), 22.5)
        assert 0.0 < result.book_prob < 1.0


# ── KellyStakingCalculator ───────────────────────────────────────────────────


class TestKellyStakingCalculator:
    def setup_method(self) -> None:
        self.calc = KellyStakingCalculator()

    def test_returns_stake_result_instance(self) -> None:
        result = self.calc.compute(_make_ev(), 1000.0)
        assert isinstance(result, StakeResult)

    def test_negative_ev_returns_zero_stake_and_fraction(self) -> None:
        result = self.calc.compute(_make_ev(ev=-0.05), 1000.0)
        assert result.stake == 0.0
        assert result.kelly_fraction == 0.0

    def test_zero_ev_returns_zero_stake(self) -> None:
        # ev=0 → kelly=0 → no bet
        ev = _make_ev(ev=0.0, fair_prob=0.5128, book_prob=0.5128)
        result = self.calc.compute(ev, 1000.0)
        assert result.stake == 0.0

    def test_positive_ev_produces_positive_stake(self) -> None:
        # ev=0.17, book_prob=0.5128 → kelly≈0.179 → stake≈179 with bankroll=1000
        result = self.calc.compute(_make_ev(ev=0.17, book_prob=0.5128), 1000.0)
        assert result.stake > 0.0
        assert result.kelly_fraction > 0.0

    def test_kelly_fraction_capped_at_thirty_percent(self) -> None:
        # ev=0.80, book_prob=0.40 → kelly=0.80*0.40/0.60≈0.533 → capped to 0.30
        result = self.calc.compute(
            _make_ev(ev=0.80, fair_prob=0.90, book_prob=0.40), 1000.0
        )
        assert result.kelly_fraction <= 0.30

    def test_stake_capped_at_thirty_percent_of_bankroll(self) -> None:
        result = self.calc.compute(
            _make_ev(ev=0.80, fair_prob=0.90, book_prob=0.40), 1000.0
        )
        assert result.stake <= 300.0

    def test_stake_below_five_euros_suppressed_to_zero(self) -> None:
        # bankroll=10, kelly≈0.179 → stake≈1.79 < €5 → suppressed
        result = self.calc.compute(_make_ev(ev=0.17, book_prob=0.5128), 10.0)
        assert result.stake == 0.0

    def test_bankroll_used_equals_passed_bankroll(self) -> None:
        result = self.calc.compute(_make_ev(), 2000.0)
        assert result.bankroll_used == 2000.0

    def test_metadata_fields_propagated_from_ev_result(self) -> None:
        result = self.calc.compute(_make_ev(), 1000.0)
        assert result.match_id == "M1"
        assert result.market == "fouls_over_under"

    def test_stake_satisfies_bankroll_invariant(self) -> None:
        result = self.calc.compute(_make_ev(), 500.0)
        assert 0.0 <= result.stake <= 500.0

    def test_kelly_formula_matches_ev_over_decimal_minus_one(self) -> None:
        # kelly = ev * book_prob / (1 - book_prob)
        # ev=0.17, book_prob=0.5128 → kelly = 0.17*0.5128/0.4872 ≈ 0.1789
        result = self.calc.compute(_make_ev(ev=0.17, book_prob=0.5128), 10_000.0)
        expected_kelly = 0.17 * 0.5128 / (1.0 - 0.5128)
        assert abs(result.kelly_fraction - expected_kelly) < 1e-6
