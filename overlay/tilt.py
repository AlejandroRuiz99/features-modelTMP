"""
overlay.tilt — P3: PMF tilt using distributions.py utilities.

Public API:
    TiltResult — dataclass holding tilted prediction dict + suppression metadata.
    apply_pmf_tilt(prediction, delta_fouls, variance_scale) -> TiltResult

Constraint (REQ-3.8): distributions.py is FROZEN — never modified.
    tilt_pmf_to_mean() and scale_pmf_variance() are called as-is.

GMM floor suppression (REQ-3.7):
    After tilt, if |realized_delta| < 0.5 * |requested_delta|, the
    GMM floor (or PMF boundary constraint) likely suppressed the shift.
    In this case TiltResult.suppressed_by_floor = True.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure prediction_models is importable
ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "prediction_models"
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from src.utils.distributions import (  # noqa: E402  # type: ignore[import]
    FoulPMF,
    scale_pmf_variance,
    tilt_pmf_to_mean,
)

__all__ = [
    "TiltResult",
    "apply_pmf_tilt",
]

# Threshold: if |realized_delta| < SUPPRESSION_THRESHOLD * |requested_delta|,
# flag suppressed_by_floor=True (REQ-3.7: "realized delta < 0.5 x requested delta").
SUPPRESSION_THRESHOLD = 0.5


@dataclass
class TiltResult:
    """Result of applying PMF tilt to a prediction.

    Attributes:
        prediction: Updated prediction dict with tilted PMF, expected_fouls,
            home_expected, away_expected, over_under (all recomputed).
        suppressed_by_floor: True if the realized shift was significantly smaller
            than the requested shift (GMM floor or boundary suppression).
        requested_delta: The delta_fouls that was requested.
        realized_delta: The actual shift achieved (post_mean - pre_mean).
    """

    prediction: dict
    suppressed_by_floor: bool
    requested_delta: float
    realized_delta: float


def apply_pmf_tilt(
    prediction: dict,
    delta_fouls: float,
    variance_scale: float,
) -> TiltResult:
    """Apply PMF tilt and variance scaling to a prediction dict.

    Steps:
      1. Compute target mean = original_mean + delta_fouls.
      2. Call tilt_pmf_to_mean(base_pmf, target_mean) → directionally shifted PMF.
      3. Call scale_pmf_variance(tilted_pmf, variance_scale) → spread-adjusted PMF.
      4. Rescale home/away expected fouls proportionally to new total mean
         (preserving the pre-tilt home/away ratio — REQ-3.6).
      5. Recompute O/U probabilities at all lines in the original over_under dict.
      6. Detect GMM floor suppression (REQ-3.7).

    Args:
        prediction: Dict with keys:
            - 'pmf_total': FoulPMF — the base PMF from ensemble.predict()
            - 'expected_fouls': float — original mean (= pmf_total.mean)
            - 'home_expected': float — home team expected fouls
            - 'away_expected': float — away team expected fouls
            - 'over_under': dict[float, tuple[float, float]] — O/U probs
        delta_fouls: Requested mean shift (positive → more fouls, negative → fewer).
        variance_scale: Variance scaling factor (>1 → wider, <1 → narrower).
            1.0 is identity for variance.

    Returns:
        TiltResult with updated prediction and suppression metadata.
    """
    base_pmf: FoulPMF = prediction["pmf_total"]
    original_mean: float = base_pmf.mean
    home_expected: float = float(prediction["home_expected"])
    away_expected: float = float(prediction["away_expected"])
    original_ou: dict = prediction.get("over_under", {})

    # Step 1: target mean
    target_mean = original_mean + delta_fouls

    # Step 2: directional tilt
    if abs(delta_fouls) < 1e-9:
        tilted_pmf = FoulPMF(probs=base_pmf.probs.copy())
    else:
        tilted_pmf = tilt_pmf_to_mean(base_pmf, target_mean)

    # Step 3: variance scaling
    if abs(variance_scale - 1.0) < 1e-9:
        final_pmf = FoulPMF(probs=tilted_pmf.probs.copy())
    else:
        final_pmf = scale_pmf_variance(tilted_pmf, variance_scale)

    new_mean = final_pmf.mean

    # Step 4: rescale home/away proportionally (preserve ratio)
    original_total = home_expected + away_expected
    if original_total > 1e-9:
        ratio_home = home_expected / original_total
        ratio_away = away_expected / original_total
    else:
        ratio_home = 0.5
        ratio_away = 0.5

    new_home = new_mean * ratio_home
    new_away = new_mean * ratio_away

    # Step 5: recompute O/U probabilities at the original lines
    lines = list(original_ou.keys()) if original_ou else None
    new_ou = final_pmf.over_under_table(lines=lines)

    # Step 6: GMM floor suppression detection (REQ-3.7)
    realized_delta = new_mean - original_mean
    suppressed = _is_suppressed(
        requested_delta=delta_fouls, realized_delta=realized_delta
    )

    new_prediction = {
        "pmf_total": final_pmf,
        "expected_fouls": new_mean,
        "home_expected": new_home,
        "away_expected": new_away,
        "over_under": new_ou,
    }

    return TiltResult(
        prediction=new_prediction,
        suppressed_by_floor=suppressed,
        requested_delta=float(delta_fouls),
        realized_delta=float(realized_delta),
    )


def _is_suppressed(requested_delta: float, realized_delta: float) -> bool:
    """Return True if realized shift is significantly smaller than requested.

    Condition (REQ-3.7): |realized_delta| < SUPPRESSION_THRESHOLD * |requested_delta|

    If requested_delta is effectively zero, no suppression is declared.
    """
    if abs(requested_delta) < 1e-9:
        return False
    return abs(realized_delta) < SUPPRESSION_THRESHOLD * abs(requested_delta)
