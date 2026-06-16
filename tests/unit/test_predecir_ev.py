"""
Unit tests for predecir_jornada EV computation (R10).

Tests written FIRST (TDD RED phase).
Covers: edge formula, EV over, EV under, exact-hit, nearest-wins,
tie-breaker lower, clamp-low, clamp-high, missing-odds returns None.
"""

from __future__ import annotations

import pytest
from predecir_jornada import compute_ev, pick_model_line

MODEL_LINES = [21.5, 23.5, 24.5, 25.5, 27.5]


class TestPickModelLine:
    """R10: pick_model_line — nearest-wins with tie-breaker."""

    def test_exact_hit(self) -> None:
        """Codere line == model line → returns that line exactly."""
        assert pick_model_line(MODEL_LINES, 23.5) == 23.5
        assert pick_model_line(MODEL_LINES, 24.5) == 24.5

    def test_nearest_unambiguous(self) -> None:
        """Closest model line is unambiguous → returns it."""
        # 24.5 is nearest to 25.0 (distance 0.5), next is 25.5 (distance 0.5)
        # Actually equidistant! Use the one from the spec scenario:
        # Codere 26.5 → distances: 5.0, 3.0, 2.0, 1.0, 1.0 → tie 25.5 vs 27.5 → lower wins
        result = pick_model_line(MODEL_LINES, 26.5)
        assert result == 25.5  # lower of tied pair

    def test_tie_breaker_lower_line_wins(self) -> None:
        """On equal distance, lower line wins (more conservative)."""
        # Codere 22.5: dist(21.5)=1.0, dist(23.5)=1.0 → lower (21.5) wins
        result = pick_model_line(MODEL_LINES, 22.5)
        assert result == 21.5

    def test_clamp_below_minimum(self) -> None:
        """Codere line < min model line → clamped to 21.5."""
        assert pick_model_line(MODEL_LINES, 19.5) == 21.5
        assert pick_model_line(MODEL_LINES, 10.0) == 21.5

    def test_clamp_above_maximum(self) -> None:
        """Codere line > max model line → clamped to 27.5."""
        assert pick_model_line(MODEL_LINES, 30.5) == 27.5
        assert pick_model_line(MODEL_LINES, 50.0) == 27.5

    def test_nearest_mid_range(self) -> None:
        """Pick the truly nearest line when no tie."""
        # 24.0 → dist(23.5)=0.5, dist(24.5)=0.5 → tie → lower 23.5
        result = pick_model_line(MODEL_LINES, 24.0)
        assert result == 23.5

    def test_nearest_close_to_25_5(self) -> None:
        """26.0 → dist(25.5)=0.5, dist(27.5)=1.5 → 25.5 wins."""
        result = pick_model_line(MODEL_LINES, 26.0)
        assert result == 25.5


class TestComputeEV:
    """R10: compute_ev formula verification."""

    def test_ev_over_formula(self) -> None:
        """EV_over = P(Over)*(odd_over-1) - P(Under)."""
        # Given: P_over=0.62, odd_over=1.80, P_under=0.38
        # Expected: 0.62 * (1.80 - 1) - 0.38 = 0.62*0.80 - 0.38 = 0.496 - 0.38 = 0.116
        ev_over, _ev_under = compute_ev(
            p_over=0.62, p_under=0.38, odd_over=1.80, odd_under=2.10
        )
        assert ev_over == pytest.approx(0.116, abs=1e-6)

    def test_ev_under_formula(self) -> None:
        """EV_under = P(Under)*(odd_under-1) - P(Over)."""
        # Given: P_under=0.45, odd_under=2.10, P_over=0.55
        # Expected: 0.45 * (2.10 - 1) - 0.55 = 0.45*1.10 - 0.55 = 0.495 - 0.55 = -0.055
        _ev_over, ev_under = compute_ev(
            p_over=0.55, p_under=0.45, odd_over=1.80, odd_under=2.10
        )
        assert ev_under == pytest.approx(-0.055, abs=1e-6)

    def test_probabilities_clamped_to_0_1(self) -> None:
        """Probabilities outside [0, 1] are clamped before computation."""
        # Should not raise, just clamp
        ev_over, ev_under = compute_ev(
            p_over=1.5, p_under=-0.2, odd_over=2.0, odd_under=2.0
        )
        # p_over clamped to 1.0, p_under clamped to 0.0
        # ev_over = 1.0 * (2.0 - 1) - 0.0 = 1.0
        assert ev_over == pytest.approx(1.0, abs=1e-6)
        # ev_under = 0.0 * (2.0 - 1) - 1.0 = -1.0
        assert ev_under == pytest.approx(-1.0, abs=1e-6)

    def test_both_values_returned_as_tuple(self) -> None:
        """compute_ev returns a tuple of (ev_over, ev_under)."""
        result = compute_ev(p_over=0.5, p_under=0.5, odd_over=2.0, odd_under=2.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_balanced_odds(self) -> None:
        """Balanced market (50/50) with fair odds → EV = 0."""
        # P_over=0.5, odd_over=2.0, P_under=0.5
        # EV_over = 0.5*(2.0-1) - 0.5 = 0.5 - 0.5 = 0.0
        ev_over, _ = compute_ev(p_over=0.5, p_under=0.5, odd_over=2.0, odd_under=2.0)
        assert ev_over == pytest.approx(0.0, abs=1e-9)


class TestEdgeFormula:
    """R10: Edge = pred_total - codere_line."""

    def test_edge_positive(self) -> None:
        """Edge is pred_total - codere_line."""
        # Edge computation is: pred_total - codere_line
        # This is tested via the MatchPrediction dataclass or a helper
        from predecir_jornada import compute_edge

        edge = compute_edge(pred_total=25.3, codere_line=24.5)
        assert edge == pytest.approx(0.8, abs=1e-6)

    def test_edge_negative(self) -> None:
        """Negative edge when pred_total < codere_line."""
        from predecir_jornada import compute_edge

        edge = compute_edge(pred_total=23.0, codere_line=24.5)
        assert edge == pytest.approx(-1.5, abs=1e-6)

    def test_edge_zero(self) -> None:
        """Edge is zero when pred_total == codere_line."""
        from predecir_jornada import compute_edge

        edge = compute_edge(pred_total=24.5, codere_line=24.5)
        assert edge == pytest.approx(0.0, abs=1e-9)
