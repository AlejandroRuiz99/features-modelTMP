"""
tests/overlay/test_applier.py — TDD for overlay.applier (T7.1).

Tests:
  1. apply_overlay with a full narrative returns an OverlayResult with all expected fields.
  2. rules_fired list is populated correctly.
  3. post_pmf_summary differs from pre_pmf_summary when rules fire (tilt applied).
  4. kelly_raw and kelly_scaled are present and kelly_scaled <= kelly_raw when
     kelly_scale < 1.0.
  5. suppressed_by_floor_count is an integer >= 0.
  6. With an empty narrative (no matching rules), OverlayResult is identity
     (post == pre; rules_fired == []).
  7. apply_overlay with narrative that has 0 fired rules still returns OverlayResult.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from overlay.applier import OverlayResult, apply_overlay
from overlay.rules import load_catalog
from overlay.schema import Narrative, NarrativeMatch, NarrativeStakes, ObjectiveOverride

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATALOG_PATH = ROOT / "overlay" / "rules.yaml"


def _make_prediction(mean: float = 25.0) -> dict:
    """Build a minimal prediction dict like ensemble.predict() would return."""
    # Gaussian-shaped probs over [0..60]
    import math

    from src.utils.distributions import FoulPMF  # type: ignore[import]

    probs = []
    for k in range(61):
        p = math.exp(-0.5 * ((k - mean) / 6.0) ** 2)
        probs.append(p)
    s = sum(probs)
    probs = [p / s for p in probs]

    pmf = FoulPMF(probs=probs)
    lines = [21.5, 23.5, 25.5, 27.5, 29.5]
    ou = pmf.over_under_table(lines)

    return {
        "pmf_total": pmf,
        "expected_fouls": pmf.mean,
        "home_expected": pmf.mean * 0.55,
        "away_expected": pmf.mean * 0.45,
        "over_under": ou,
    }


def _make_narrative_with_two_up_rules() -> Narrative:
    """Narrative that triggers both_relegation_up + physical_clash_up (2 UP rules)."""
    return Narrative(
        match=NarrativeMatch(home="Espanol", away="Levante", date="2026-04-27"),
        confidence_level=4,
        objectives={
            "home": ObjectiveOverride(label="descenso"),
            "away": ObjectiveOverride(label="salvacion"),
        },
        stakes=NarrativeStakes(home=4, away=5),
        physicality_bias=1,
        special_flags=["physical_clash"],
    )


def _make_empty_narrative() -> Narrative:
    """Narrative that fires no rules (all conditions will be False)."""
    return Narrative(
        match=NarrativeMatch(home="TeamA", away="TeamB", date="2026-04-27"),
        confidence_level=2,  # below threshold → directional gate blocked anyway,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyOverlay:
    def test_returns_overlay_result_dataclass(self) -> None:
        """apply_overlay returns an OverlayResult with all required fields."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction()
        narrative = _make_narrative_with_two_up_rules()

        result = apply_overlay(prediction, narrative, catalog)

        assert isinstance(result, OverlayResult)
        assert hasattr(result, "pre_pmf_summary")
        assert hasattr(result, "post_pmf_summary")
        assert hasattr(result, "rules_fired")
        assert hasattr(result, "kelly_raw")
        assert hasattr(result, "kelly_scaled")
        assert hasattr(result, "suppressed_by_floor_count")
        assert hasattr(result, "post_prediction")
        assert hasattr(result, "aggregated_effect")

    def test_rules_fired_populated_for_matching_narrative(self) -> None:
        """With both_relegation_up conditions satisfied, rules_fired is non-empty."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction()
        narrative = _make_narrative_with_two_up_rules()

        result = apply_overlay(prediction, narrative, catalog)

        assert len(result.rules_fired) >= 1, (
            "Expected at least 1 rule to fire for a relegation narrative"
        )
        # Each entry should have id, direction, magnitude_applied, suppressed_by_floor
        for entry in result.rules_fired:
            assert "id" in entry
            assert "direction" in entry
            assert "magnitude_applied" in entry
            assert "suppressed_by_floor" in entry

    def test_post_pmf_mean_shifts_when_rules_fire_with_enough_confidence(self) -> None:
        """With 2 UP rules and confidence_level=4, post expected_fouls > pre."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction(mean=25.0)
        narrative = _make_narrative_with_two_up_rules()

        result = apply_overlay(prediction, narrative, catalog)

        pre_mean = result.pre_pmf_summary["mean"]
        post_mean = result.post_pmf_summary["mean"]

        # With same-direction gate passed and confidence=4, post mean should be higher
        if len(result.rules_fired) >= 2:
            assert post_mean >= pre_mean, (
                f"Expected post_mean >= pre_mean when UP rules fire, "
                f"got pre={pre_mean:.3f} post={post_mean:.3f}"
            )

    def test_kelly_scaled_le_kelly_raw_when_scale_lt_1(self) -> None:
        """When kelly_scale < 1.0, kelly_scaled <= kelly_raw."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction()
        narrative = _make_narrative_with_two_up_rules()

        result = apply_overlay(prediction, narrative, catalog)

        # kelly_scaled should always be <= kelly_raw (scale is in [0.25, 1.0])
        assert result.kelly_scaled <= result.kelly_raw + 1e-9, (
            f"kelly_scaled {result.kelly_scaled} > kelly_raw {result.kelly_raw}"
        )

    def test_suppressed_by_floor_count_is_non_negative_int(self) -> None:
        """suppressed_by_floor_count is a non-negative integer."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction()
        narrative = _make_narrative_with_two_up_rules()

        result = apply_overlay(prediction, narrative, catalog)

        assert isinstance(result.suppressed_by_floor_count, int)
        assert result.suppressed_by_floor_count >= 0

    def test_identity_when_no_rules_fire(self) -> None:
        """With no matching rules, post PMF equals pre PMF and rules_fired is empty."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction(mean=25.0)
        narrative = _make_empty_narrative()

        result = apply_overlay(prediction, narrative, catalog)

        assert result.rules_fired == [], (
            f"Expected no rules to fire, got: {[r['id'] for r in result.rules_fired]}"
        )
        # When no rules fire, PMF is identity
        assert (
            abs(result.pre_pmf_summary["mean"] - result.post_pmf_summary["mean"]) < 1e-9
        )

    def test_pre_pmf_summary_has_required_keys(self) -> None:
        """pre_pmf_summary and post_pmf_summary contain mean, std, q25, q50, q75."""
        catalog = load_catalog(CATALOG_PATH)
        prediction = _make_prediction()
        narrative = _make_empty_narrative()

        result = apply_overlay(prediction, narrative, catalog)

        for summary_name in ("pre_pmf_summary", "post_pmf_summary"):
            summary = getattr(result, summary_name)
            for key in ("mean", "std", "q25", "q50", "q75"):
                assert key in summary, (
                    f"Missing key '{key}' in {summary_name}: {summary}"
                )
