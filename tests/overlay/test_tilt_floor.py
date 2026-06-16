"""
tests/overlay/test_tilt_floor.py — Floor suppression detection for overlay.tilt.

Tests (2 cases):
  1. Downward tilt below the PMF boundary (low mean) triggers suppressed_by_floor=True.
     Simulates the ensemble GMM floor scenario by requesting a shift that cannot be
     fully realized because the PMF hits the distribution boundary.
  2. A tilt that IS fully realized (no boundary hit) → suppressed_by_floor=False.

The GMM floor in ensemble.py (lines 490-495) raises the effective mean when home/away
fouls_committed_avg < 11.5. In overlay/tilt.py we detect this by comparing realized
vs requested delta: if |realized| < 0.5 * |requested|, flag suppressed_by_floor=True.

These tests work at the tilt module level (no ensemble call needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from overlay.tilt import apply_pmf_tilt
from src.utils.distributions import pmf_from_negbin


class TestGmmFloorSuppression:
    def test_large_downward_tilt_on_very_low_mean_triggers_suppression(self) -> None:
        """Requesting -5.0 fouls on PMF with mean=1.5 → suppressed_by_floor=True.

        A PMF with mean=1.5 can only shift down to ~0 (physical lower bound).
        So the realized delta is at most ~-1.5, which is < 0.5 * 5.0 = 2.5,
        triggering suppressed_by_floor=True.

        This simulates the ensemble GMM floor scenario: when the PMF hits its
        minimum boundary and cannot achieve the requested downward tilt.
        """
        pmf_very_low = pmf_from_negbin(mu=1.5, alpha=0.3)
        prediction = {
            "pmf_total": pmf_very_low,
            "expected_fouls": pmf_very_low.mean,
            "home_expected": 0.75,
            "away_expected": 0.75,
            "over_under": pmf_very_low.over_under_table(),
        }

        result = apply_pmf_tilt(prediction, delta_fouls=-5.0, variance_scale=1.0)

        # From mean ~1.5, can only shift down to ~0: |realized| <= 1.5 < 2.5 (= 0.5 * 5.0)
        assert abs(result.realized_delta) < 2.5, (
            f"Expected |realized_delta| < 2.5, got {result.realized_delta:.3f}"
        )
        # suppressed_by_floor must be True (|realized| < 0.5 * |requested|)
        assert result.suppressed_by_floor is True, (
            f"Expected suppressed_by_floor=True, got False "
            f"(realized={result.realized_delta:.3f}, requested={result.requested_delta:.3f})"
        )
        # Final mean must be >= 0 (physically bounded)
        final_mean = result.prediction["pmf_total"].mean
        assert final_mean >= 0.0, f"PMF mean must be non-negative, got {final_mean:.3f}"

    def test_realizable_tilt_does_not_trigger_suppression(self) -> None:
        """A small upward tilt on a normal-range PMF → suppressed_by_floor=False.

        Requesting +1.0 fouls on PMF with mean=25.0 is easily achievable.
        realized_delta should be >= 0.5 * 1.0 = 0.5, so no suppression.
        """
        pmf_normal = pmf_from_negbin(mu=25.0, alpha=0.3)
        prediction = {
            "pmf_total": pmf_normal,
            "expected_fouls": pmf_normal.mean,
            "home_expected": 13.0,
            "away_expected": 12.0,
            "over_under": pmf_normal.over_under_table(),
        }

        result = apply_pmf_tilt(prediction, delta_fouls=1.0, variance_scale=1.0)

        # realized_delta should be >= 0.5 (half of requested 1.0)
        assert abs(result.realized_delta) >= 0.5, (
            f"Expected |realized_delta| >= 0.5, got {result.realized_delta:.3f}"
        )
        # suppressed_by_floor must be False
        assert result.suppressed_by_floor is False, (
            f"Expected suppressed_by_floor=False, got True "
            f"(realized={result.realized_delta:.3f}, requested={result.requested_delta:.3f})"
        )
