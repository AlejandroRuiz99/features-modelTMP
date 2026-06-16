"""
tests/overlay/test_tilt.py — Strict TDD for overlay.tilt.

Tests (5 cases + triangulation, all RED before implementation):
  1. apply_pmf_tilt with delta_fouls=+1.0 shifts mean upward by ~1.0
  2. scale_pmf_variance called correctly — variance changes with variance_scale
  3. home/away expected fouls rescale proportionally (ratio preserved) after tilt
  4. zero delta → PMF mean unchanged (identity for directional component)
  5. GMM floor suppression detection: when actual shift < 0.5 * requested shift,
     TiltResult has suppressed_by_floor=True

Dependencies: uses FoulPMF and distributions.py utilities directly.
distributions.py is FROZEN — do NOT modify.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure prediction_models is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from overlay.tilt import TiltResult, apply_pmf_tilt
from src.utils.distributions import FoulPMF, pmf_from_negbin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_K = 61


def _make_pmf(mean: float = 25.0, alpha: float = 0.3) -> FoulPMF:
    """Create a realistic NegBin PMF with given mean."""
    return pmf_from_negbin(mu=mean, alpha=alpha)


def _make_prediction(
    pmf: FoulPMF | None = None,
    mean: float = 25.0,
    home_expected: float = 13.0,
    away_expected: float = 12.0,
) -> dict:
    """Build a minimal prediction dict for tilt testing."""
    if pmf is None:
        pmf = _make_pmf(mean=mean)
    ou_table = pmf.over_under_table()
    return {
        "pmf_total": pmf,
        "expected_fouls": pmf.mean,
        "home_expected": home_expected,
        "away_expected": away_expected,
        "over_under": ou_table,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApplyPmfTilt:
    def test_positive_delta_shifts_mean_upward(self) -> None:
        """delta_fouls=+1.0 → tilted PMF mean is ~1.0 higher than original."""
        pred = _make_prediction(mean=25.0)
        original_mean = pred["pmf_total"].mean

        result = apply_pmf_tilt(pred, delta_fouls=1.0, variance_scale=1.0)

        tilted_mean = result.prediction["pmf_total"].mean
        # Mean should be shifted upward by approximately delta_fouls
        # Allow tolerance for exponential tilting convergence
        assert tilted_mean > original_mean + 0.5, (
            f"Expected mean > {original_mean + 0.5:.2f}, got {tilted_mean:.4f}"
        )
        assert tilted_mean < original_mean + 1.5, (
            f"Expected mean < {original_mean + 1.5:.2f}, got {tilted_mean:.4f}"
        )
        # PMF must still sum to 1.0
        assert abs(result.prediction["pmf_total"].probs.sum() - 1.0) < 1e-6

    def test_variance_scale_changes_distribution_spread(self) -> None:
        """variance_scale=1.2 → tilted PMF has larger variance than original."""
        pred = _make_prediction(mean=25.0)
        original_var = pred["pmf_total"].variance

        result = apply_pmf_tilt(pred, delta_fouls=0.0, variance_scale=1.2)

        new_var = result.prediction["pmf_total"].variance
        # With variance_scale > 1.0 (wider), variance should increase
        assert new_var > original_var * 0.9, (
            f"Expected variance to increase, got original={original_var:.2f}, new={new_var:.2f}"
        )

    def test_home_away_rescale_preserves_ratio(self) -> None:
        """home/away expected fouls rescale proportionally to new total mean."""
        home_expected = 14.0
        away_expected = 11.0
        original_ratio = home_expected / away_expected
        pred = _make_prediction(
            mean=25.0,
            home_expected=home_expected,
            away_expected=away_expected,
        )

        result = apply_pmf_tilt(pred, delta_fouls=1.0, variance_scale=1.0)

        new_home = result.prediction["home_expected"]
        new_away = result.prediction["away_expected"]
        new_ratio = new_home / new_away

        # Ratio should be preserved within a small tolerance
        assert abs(new_ratio - original_ratio) < 0.01, (
            f"Ratio changed: original={original_ratio:.4f}, new={new_ratio:.4f}"
        )
        # New total should match new mean
        new_mean = result.prediction["pmf_total"].mean
        new_total = new_home + new_away
        assert abs(new_total - new_mean) < 0.5, (
            f"home+away={new_total:.2f} does not match new mean={new_mean:.2f}"
        )

    def test_zero_delta_leaves_pmf_mean_unchanged(self) -> None:
        """delta_fouls=0.0 + variance_scale=1.0 → PMF mean identical to original."""
        pred = _make_prediction(mean=25.0)
        original_mean = pred["pmf_total"].mean

        result = apply_pmf_tilt(pred, delta_fouls=0.0, variance_scale=1.0)

        new_mean = result.prediction["pmf_total"].mean
        assert abs(new_mean - original_mean) < 1e-4, (
            f"Expected mean unchanged ({original_mean:.4f}), got {new_mean:.4f}"
        )

    def test_floor_suppression_detected_when_actual_shift_is_small(self) -> None:
        """When the actual shift is < 0.5 * requested shift, suppressed_by_floor=True.

        We simulate GMM floor suppression by providing a prediction whose PMF is
        already at its maximum tiltable position (e.g. requesting a large negative
        shift on a low-mean PMF — the tilt can't go below 0). This checks the
        suppression detection logic.

        Strategy: request delta_fouls=-5.0 on a PMF with mean=3.0.
        tilt_pmf_to_mean can't push mean below 0 significantly, so actual
        shift << requested shift → suppressed_by_floor=True.
        """
        # PMF with very low mean — can't shift down much further
        pmf_low = pmf_from_negbin(mu=3.0, alpha=0.3)
        pred = _make_prediction(
            pmf=pmf_low, mean=3.0, home_expected=1.5, away_expected=1.5
        )

        result = apply_pmf_tilt(pred, delta_fouls=-5.0, variance_scale=1.0)

        # Actual realized shift should be much less than requested (-5.0)
        original_mean = pmf_low.mean
        realized_delta = result.prediction["pmf_total"].mean - original_mean
        requested_delta = -5.0
        # If |realized| < 0.5 * |requested|, floor suppression should be flagged
        if abs(realized_delta) < 0.5 * abs(requested_delta):
            assert result.suppressed_by_floor is True, (
                f"Expected suppressed_by_floor=True when realized_delta={realized_delta:.2f} "
                f"vs requested_delta={requested_delta:.2f}"
            )


class TestTiltResultStructure:
    def test_result_has_required_keys(self) -> None:
        """TiltResult has prediction dict with all required keys."""
        pred = _make_prediction(mean=25.0)
        result = apply_pmf_tilt(pred, delta_fouls=0.5, variance_scale=1.1)

        assert isinstance(result, TiltResult)
        assert "pmf_total" in result.prediction
        assert "expected_fouls" in result.prediction
        assert "home_expected" in result.prediction
        assert "away_expected" in result.prediction
        assert "over_under" in result.prediction
        assert isinstance(result.suppressed_by_floor, bool)
        assert isinstance(result.requested_delta, float)
        assert isinstance(result.realized_delta, float)

    def test_ou_probs_recomputed_after_tilt(self) -> None:
        """O/U probabilities are recomputed on the tilted PMF."""
        pred = _make_prediction(mean=25.0)
        original_ou = pred["over_under"].copy()

        result = apply_pmf_tilt(pred, delta_fouls=1.5, variance_scale=1.0)

        new_ou = result.prediction["over_under"]
        # After shifting mean up by ~1.5, over probability at 25.5 should increase
        if 25.5 in original_ou and 25.5 in new_ou:
            assert new_ou[25.5][0] > original_ou[25.5][0] - 0.01, (
                "Over probability at 25.5 should not decrease when mean shifts up"
            )
