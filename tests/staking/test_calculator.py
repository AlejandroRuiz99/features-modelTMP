"""
tests/staking/test_calculator.py — TDD for staking.calculator

Tests (T3.4 / T3.3 in batch task numbering):
  1. D6 worked example: edge=0.12, kelly=0.85, bankroll=500 → euros=153.
  2. Edge below no_bet threshold → stake_n=0, euros=0.
  3. Edge above max threshold → stake_n=10 (maximum).
  4. StakeResult has all expected fields.
  5. kelly_scale applied BEFORE euro quantize (D6 timing).
  6. Edge exactly on boundary → takes the correct tier.
"""

from __future__ import annotations

import pytest

from staking.calculator import compute_stake
from staking.loader import StakingCurve


def _make_default_curve() -> StakingCurve:
    """Build the default staking curve matching overlay/staking.yaml."""
    return StakingCurve(
        bankroll_share_per_stake_unit=0.06,
        no_bet_below_edge=0.05,
        edge_thresholds=[
            {"edge_min": 0.05, "edge_max": 0.07, "stake": 2},
            {"edge_min": 0.07, "edge_max": 0.10, "stake": 4},
            {"edge_min": 0.10, "edge_max": 0.13, "stake": 6},
            {"edge_min": 0.13, "edge_max": 0.16, "stake": 8},
            {"edge_min": 0.16, "edge_max": 1.00, "stake": 10},
        ],
    )


class TestD6WorkedExample:
    def test_d6_worked_example(self) -> None:
        """D6 example: edge=0.12, kelly=0.85, bankroll=500 → euros≈153."""
        # edge=0.12 → tier [0.10, 0.13) → stake_n=6
        # bank_share_raw = 6 * 0.06 = 0.36
        # bank_share_final = 0.36 * 0.85 = 0.306
        # euros = 0.306 * 500 = 153.0
        curve = _make_default_curve()
        result = compute_stake(edge=0.12, kelly_scale=0.85, bankroll=500.0, curve=curve)

        assert result.stake_n == 6
        assert result.bank_share_raw == pytest.approx(0.36)
        assert result.bank_share_final == pytest.approx(0.306)
        assert result.euros == pytest.approx(153.0)

    def test_kelly_scale_is_stored(self) -> None:
        """StakeResult stores the kelly_scale used."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.12, kelly_scale=0.85, bankroll=500.0, curve=curve)
        assert result.kelly_scale == pytest.approx(0.85)

    def test_edge_is_stored(self) -> None:
        """StakeResult stores the input edge."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.12, kelly_scale=0.85, bankroll=500.0, curve=curve)
        assert result.edge == pytest.approx(0.12)


class TestEdgeBelowThreshold:
    def test_edge_below_no_bet_returns_zero(self) -> None:
        """Edge below no_bet_below_edge → stake_n=0, euros=0."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.03, kelly_scale=1.0, bankroll=500.0, curve=curve)
        assert result.stake_n == 0
        assert result.euros == 0.0

    def test_edge_exactly_at_no_bet_boundary(self) -> None:
        """Edge exactly at no_bet_below_edge is still no-bet (below or equal)."""
        curve = _make_default_curve()
        # no_bet_below_edge=0.05 → edge=0.05 is the MINIMUM for betting
        # The first tier starts at edge_min=0.05, so 0.05 should get stake_n=2
        result = compute_stake(edge=0.05, kelly_scale=1.0, bankroll=500.0, curve=curve)
        # edge=0.05 is the edge_min of tier 1 → stake_n=2
        assert result.stake_n == 2


class TestEdgeAboveMaxThreshold:
    def test_edge_above_max_returns_max_stake(self) -> None:
        """Edge above all thresholds → max stake_n=10."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.50, kelly_scale=1.0, bankroll=500.0, curve=curve)
        assert result.stake_n == 10

    def test_edge_at_upper_tier(self) -> None:
        """Edge at 0.20 (in top tier ≥0.16) → stake_n=10."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.20, kelly_scale=1.0, bankroll=500.0, curve=curve)
        assert result.stake_n == 10


class TestStakeResultFields:
    def test_all_expected_fields_present(self) -> None:
        """StakeResult has edge, stake_n, bank_share_raw, kelly_scale, bank_share_final, euros."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.08, kelly_scale=1.0, bankroll=1000.0, curve=curve)

        assert hasattr(result, "edge")
        assert hasattr(result, "stake_n")
        assert hasattr(result, "bank_share_raw")
        assert hasattr(result, "kelly_scale")
        assert hasattr(result, "bank_share_final")
        assert hasattr(result, "euros")

    def test_mid_tier_calculation(self) -> None:
        """edge=0.08 → tier [0.07, 0.10) → stake_n=4."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.08, kelly_scale=1.0, bankroll=1000.0, curve=curve)
        assert result.stake_n == 4
        assert result.bank_share_raw == pytest.approx(0.24)  # 4 * 0.06
        assert result.bank_share_final == pytest.approx(0.24)  # kelly=1.0
        assert result.euros == pytest.approx(240.0)  # 0.24 * 1000

    def test_kelly_scale_applied_before_euros(self) -> None:
        """D6: kelly_scale multiplies bank_share, not stake_n."""
        curve = _make_default_curve()
        # With kelly=0.5
        result = compute_stake(edge=0.08, kelly_scale=0.5, bankroll=1000.0, curve=curve)
        assert result.stake_n == 4  # stake_n unchanged by kelly_scale
        assert result.bank_share_final == pytest.approx(0.12)  # 0.24 * 0.5
        assert result.euros == pytest.approx(120.0)  # 0.12 * 1000


class TestBoundaryEdgeCases:
    def test_edge_at_tier_boundary_goes_to_correct_tier(self) -> None:
        """Edge exactly at a tier boundary goes to the appropriate tier."""
        curve = _make_default_curve()
        # edge=0.07 → matches edge_min=0.07 of second tier [0.07, 0.10) → stake_n=4
        result = compute_stake(edge=0.07, kelly_scale=1.0, bankroll=100.0, curve=curve)
        assert result.stake_n == 4

    def test_zero_bankroll_gives_zero_euros(self) -> None:
        """Zero bankroll → zero euros regardless of edge."""
        curve = _make_default_curve()
        result = compute_stake(edge=0.12, kelly_scale=1.0, bankroll=0.0, curve=curve)
        assert result.euros == 0.0
        assert result.stake_n == 6  # stake_n still determined by edge
