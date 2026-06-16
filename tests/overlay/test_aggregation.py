"""
tests/overlay/test_aggregation.py — Unit tests for effect aggregation.

Tests (8 cases — all GREEN since aggregate_effects was implemented in T2.2):
  1. delta_fouls sum clipped at +2.0
  2. delta_fouls sum clipped at -2.0
  3. variance_scale product clipped to [1/1.5, 1.5]
  4. kelly_scale min clipped to [0.25, 1.0]
  5. Directional gate passes when confidence>=3 and >=2 same-direction rules
  6. Gate blocks when confidence<3
  7. Gate blocks when only 1 rule has a direction (single directional rule)
  8. variance/kelly still apply when gate blocks delta
"""

from __future__ import annotations

import pytest

from overlay.rules import Rule, RuleEffect, aggregate_effects

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rule(
    delta_fouls: float = 0.0,
    variance_scale: float = 1.0,
    kelly_scale: float = 1.0,
    direction: str = "up",
    enabled: bool = True,
) -> Rule:
    return Rule(
        id=f"rule_{delta_fouls}",
        description="test",
        enabled=enabled,
        direction=direction,
        when={},
        effect=RuleEffect(
            delta_fouls=delta_fouls,
            variance_scale=variance_scale,
            kelly_scale=kelly_scale,
        ),
    )


# ---------------------------------------------------------------------------
# Case 1: delta_fouls sum clipped at +2.0
# ---------------------------------------------------------------------------


class TestDeltaFoulsCap:
    def test_sum_clipped_at_positive_cap(self) -> None:
        """3 rules each +0.8 → sum=2.4, clipped to +2.0."""
        rules = [_make_rule(delta_fouls=0.8) for _ in range(3)]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.raw_delta_sum == pytest.approx(2.4)
        assert result.delta_fouls == pytest.approx(2.0)
        assert result.directional_gate_passed is True

    def test_sum_clipped_at_negative_cap(self) -> None:
        """3 rules each -0.8 → sum=-2.4, clipped to -2.0."""
        rules = [_make_rule(delta_fouls=-0.8, direction="down") for _ in range(3)]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.raw_delta_sum == pytest.approx(-2.4)
        assert result.delta_fouls == pytest.approx(-2.0)
        assert result.directional_gate_passed is True

    def test_sum_within_cap_unclipped(self) -> None:
        """2 rules each +0.5 → sum=1.0, no clipping."""
        rules = [_make_rule(delta_fouls=0.5) for _ in range(2)]
        result = aggregate_effects(rules, confidence_level=3)
        assert result.delta_fouls == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 2: variance_scale product clipped to [1/1.5, 1.5]
# ---------------------------------------------------------------------------


class TestVarianceScaleProduct:
    def test_product_clipped_at_max(self) -> None:
        """1.4 * 1.3 = 1.82 → clipped to 1.5."""
        rules = [
            _make_rule(variance_scale=1.4),
            _make_rule(variance_scale=1.3),
        ]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.variance_scale == pytest.approx(1.5)

    def test_single_variance_no_clip(self) -> None:
        """Single rule variance_scale=1.2 → 1.2 (no clipping)."""
        rules = [_make_rule(variance_scale=1.2)]
        result = aggregate_effects(rules, confidence_level=4)
        # Note: single directional rule → gate blocks delta, but variance still applies
        assert result.variance_scale == pytest.approx(1.2)

    def test_default_variance_is_identity(self) -> None:
        """Rules with variance_scale=1.0 → product=1.0."""
        rules = [_make_rule(delta_fouls=0.5, variance_scale=1.0) for _ in range(2)]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.variance_scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Case 3: kelly_scale min clipped to [0.25, 1.0]
# ---------------------------------------------------------------------------


class TestKellyScaleMin:
    def test_min_aggregation(self) -> None:
        """min(0.9, 0.7, 0.6) = 0.6."""
        rules = [
            _make_rule(delta_fouls=0.5, kelly_scale=0.9),
            _make_rule(delta_fouls=0.5, kelly_scale=0.7),
            _make_rule(delta_fouls=0.5, kelly_scale=0.6),
        ]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.kelly_scale == pytest.approx(0.6)

    def test_kelly_clipped_at_floor(self) -> None:
        """min(0.3, 0.25) = 0.25 → already at floor."""
        rules = [
            _make_rule(delta_fouls=0.5, kelly_scale=0.3),
            _make_rule(delta_fouls=0.5, kelly_scale=0.25),
        ]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.kelly_scale == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Case 4: Directional gate passes when confidence>=3 and >=2 same-direction
# ---------------------------------------------------------------------------


class TestDirectionalGatePasses:
    def test_gate_passes_with_2_same_direction(self) -> None:
        """confidence=4, 2 up rules → gate passes, delta applied."""
        rules = [
            _make_rule(delta_fouls=0.5),
            _make_rule(delta_fouls=0.6),
        ]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.directional_gate_passed is True
        assert result.delta_fouls > 0.0

    def test_gate_passes_at_threshold_confidence(self) -> None:
        """confidence=3 (minimum) with 2 same-direction rules → gate passes."""
        rules = [_make_rule(delta_fouls=0.5) for _ in range(2)]
        result = aggregate_effects(rules, confidence_level=3)
        assert result.directional_gate_passed is True


# ---------------------------------------------------------------------------
# Case 5: Gate blocks when confidence<3
# ---------------------------------------------------------------------------


class TestDirectionalGateBlockedLowConfidence:
    def test_gate_blocked_confidence_2(self) -> None:
        """confidence=2 < 3 → gate blocked, delta=0."""
        rules = [_make_rule(delta_fouls=0.5) for _ in range(3)]
        result = aggregate_effects(rules, confidence_level=2)
        assert result.directional_gate_passed is False
        assert result.delta_fouls == pytest.approx(0.0)

    def test_gate_blocked_confidence_1(self) -> None:
        """confidence=1 → gate blocked."""
        rules = [_make_rule(delta_fouls=0.8) for _ in range(2)]
        result = aggregate_effects(rules, confidence_level=1)
        assert result.delta_fouls == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Case 6: Gate blocks when only 1 rule with direction (single directional rule)
# ---------------------------------------------------------------------------


class TestDirectionalGateBlockedSingleRule:
    def test_single_directional_rule_blocked(self) -> None:
        """Only 1 rule fires with delta → gate blocked (< 2 same-sign rules)."""
        rules = [_make_rule(delta_fouls=0.8)]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.directional_gate_passed is False
        assert result.delta_fouls == pytest.approx(0.0)

    def test_opposite_direction_rules_blocked(self) -> None:
        """1 up, 1 down of equal magnitude → sum=0, gate blocked."""
        rules = [
            _make_rule(delta_fouls=0.5, direction="up"),
            _make_rule(delta_fouls=-0.5, direction="down"),
        ]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.delta_fouls == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Case 7: variance/kelly still apply when gate blocks delta
# ---------------------------------------------------------------------------


class TestVarianceKellyApplyWhenGateBlocks:
    def test_variance_applies_single_rule(self) -> None:
        """Single directional rule: gate blocks delta but variance still scales."""
        rules = [_make_rule(delta_fouls=0.8, variance_scale=1.2, kelly_scale=0.85)]
        result = aggregate_effects(rules, confidence_level=4)
        assert result.delta_fouls == pytest.approx(0.0)  # blocked
        assert result.variance_scale == pytest.approx(1.2)  # still applies
        assert result.kelly_scale == pytest.approx(0.85)  # still applies

    def test_variance_applies_low_confidence(self) -> None:
        """Low confidence blocks delta but variance and kelly still apply."""
        rules = [
            _make_rule(delta_fouls=0.5, variance_scale=1.15, kelly_scale=0.80),
            _make_rule(delta_fouls=0.5, variance_scale=1.10, kelly_scale=0.90),
        ]
        result = aggregate_effects(rules, confidence_level=2)
        assert result.delta_fouls == pytest.approx(0.0)
        # product: 1.15 * 1.10 = 1.265
        assert result.variance_scale == pytest.approx(1.15 * 1.10)
        # min: 0.80
        assert result.kelly_scale == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Case 8: Empty fired_rules → identity
# ---------------------------------------------------------------------------


class TestEmptyFiredRules:
    def test_no_rules_returns_identity(self) -> None:
        """No fired rules → delta=0, variance=1.0, kelly=1.0."""
        result = aggregate_effects([], confidence_level=4)
        assert result.delta_fouls == pytest.approx(0.0)
        assert result.variance_scale == pytest.approx(1.0)
        assert result.kelly_scale == pytest.approx(1.0)
        assert result.directional_gate_passed is False


# ---------------------------------------------------------------------------
# Volatility rules: variance/kelly apply, delta stays 0, do NOT contribute
# to the directional gate count (since delta_fouls = 0.0 for volatility rules)
# ---------------------------------------------------------------------------


class TestVolatilityRulesAggregation:
    def test_volatility_rule_alone_has_zero_delta(self) -> None:
        """A volatility rule with delta_fouls=0.0 alone produces zero delta (gate N/A)."""
        volatility_rule = _make_rule(
            delta_fouls=0.0,
            variance_scale=1.15,
            kelly_scale=0.90,
            direction="volatility",
        )
        result = aggregate_effects([volatility_rule], confidence_level=4)
        assert result.delta_fouls == pytest.approx(0.0)
        assert result.variance_scale == pytest.approx(1.15)
        assert result.kelly_scale == pytest.approx(0.90)

    def test_volatility_rule_does_not_count_toward_directional_gate(self) -> None:
        """Volatility rule (delta=0) paired with a single UP rule does NOT satisfy
        the ≥2 same-direction gate — delta must stay blocked.

        This verifies that volatility rules with delta_fouls=0 do not masquerade as
        directional rules to inflate the same-direction count.
        """
        up_rule = _make_rule(
            delta_fouls=0.5, variance_scale=1.10, kelly_scale=0.90, direction="up"
        )
        volatility_rule = _make_rule(
            delta_fouls=0.0,
            variance_scale=1.15,
            kelly_scale=0.90,
            direction="volatility",
        )
        result = aggregate_effects([up_rule, volatility_rule], confidence_level=4)
        # Only 1 UP rule has non-zero delta → gate should block delta
        assert result.delta_fouls == pytest.approx(0.0), (
            "Single UP rule + volatility should NOT pass the directional gate"
        )
        # But variance and kelly still apply
        # product: 1.10 * 1.15 = 1.265
        assert result.variance_scale == pytest.approx(1.10 * 1.15)
        # min: 0.90
        assert result.kelly_scale == pytest.approx(0.90)

    def test_volatility_rule_variance_stacks_with_directional(self) -> None:
        """Two UP rules (gate passes) + volatility rule: variance stacks multiplicatively."""
        up1 = _make_rule(
            delta_fouls=0.5, variance_scale=1.10, kelly_scale=0.90, direction="up"
        )
        up2 = _make_rule(
            delta_fouls=0.4, variance_scale=1.05, kelly_scale=0.95, direction="up"
        )
        volatility_rule = _make_rule(
            delta_fouls=0.0,
            variance_scale=1.15,
            kelly_scale=0.90,
            direction="volatility",
        )
        result = aggregate_effects([up1, up2, volatility_rule], confidence_level=4)
        # Gate passes (2 UP rules)
        assert result.delta_fouls == pytest.approx(0.9)
        # variance: product of all three 1.10 * 1.05 * 1.15 ≈ 1.329
        expected_variance = 1.10 * 1.05 * 1.15
        assert result.variance_scale == pytest.approx(expected_variance, abs=1e-3)
        # kelly: min(0.90, 0.95, 0.90) = 0.90
        assert result.kelly_scale == pytest.approx(0.90)
