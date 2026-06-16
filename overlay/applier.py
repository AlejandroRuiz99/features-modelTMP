"""
overlay.applier — Orchestrates the post-prediction overlay pipeline (P3 + P4).

Public API:
    OverlayResult  — dataclass holding all overlay outputs.
    apply_overlay(prediction, narrative, catalog) -> OverlayResult

Pipeline:
    1. Evaluate all rules against the narrative → list of fired rules.
    2. Aggregate effects (delta_fouls, variance_scale, kelly_scale).
    3. Apply PMF tilt via overlay.tilt.apply_pmf_tilt() (P3).
    4. Derive kelly_raw / kelly_scaled from the aggregated kelly_scale (P4).
    5. Return OverlayResult with pre/post PMF summaries, fired rules list,
       and aggregation metadata.

Note on P1 (objective override):
    P1 is applied BEFORE feature generation in run_prediction.py.  This module
    only handles the POST-prediction pipeline (P3 + P4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from overlay.rules import (
    AggregatedEffect,
    Rule,
    aggregate_effects,
    evaluate_rule,
)
from overlay.schema import Narrative
from overlay.tilt import TiltResult, apply_pmf_tilt

__all__ = [
    "OverlayResult",
    "apply_overlay",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OverlayResult:
    """Complete result of the post-prediction overlay pass.

    Attributes:
        pre_pmf_summary:      PMF stats before tilt {mean, std, q25, q50, q75}.
        post_pmf_summary:     PMF stats after tilt.
        post_prediction:      Updated prediction dict (pmf_total, expected_fouls,
                              home_expected, away_expected, over_under).
        rules_fired:          List of dicts — one per fired rule:
                                {id, direction, magnitude_applied, suppressed_by_floor}
        aggregated_effect:    Full AggregatedEffect from rules.aggregate_effects().
        kelly_raw:            Kelly fraction from the prediction EV (not scaled).
                              NOTE: this is the overlay-level kelly_scale multiplier,
                              not a computed EV figure.  run_prediction.py passes
                              this to ev.py at call time.
        kelly_scaled:         kelly_raw * aggregated_effect.kelly_scale.
        suppressed_by_floor_count: Number of rules whose downward tilt was
                              suppressed by the GMM floor.
        tilt_result:          Raw TiltResult from apply_pmf_tilt().
    """

    pre_pmf_summary: dict
    post_pmf_summary: dict
    post_prediction: dict
    rules_fired: list[dict]
    aggregated_effect: AggregatedEffect
    kelly_raw: float
    kelly_scaled: float
    suppressed_by_floor_count: int
    tilt_result: TiltResult | None = field(default=None)


# ---------------------------------------------------------------------------
# PMF summary helper
# ---------------------------------------------------------------------------


def _pmf_summary(prediction: dict) -> dict:
    """Compute a summary dict from a prediction dict.

    Keys: mean, std, q25, q50, q75.
    """
    pmf = prediction["pmf_total"]
    probs = pmf.probs
    mean_val = float(pmf.mean)

    # Variance / std — E[(X-mu)^2]
    variance = sum(((i - mean_val) ** 2) * float(p) for i, p in enumerate(probs))
    std_val = variance**0.5

    # Quantiles via CDF
    q25 = _quantile(probs, 0.25)
    q50 = _quantile(probs, 0.50)
    q75 = _quantile(probs, 0.75)

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "q25": q25,
        "q50": q50,
        "q75": q75,
    }


def _quantile(probs: list[float], q: float) -> float:
    """Return the q-th quantile of a PMF over [0..len(probs)-1]."""
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += float(p)
        if cumulative >= q:
            return float(i)
    return float(len(probs) - 1)


# ---------------------------------------------------------------------------
# Core applier
# ---------------------------------------------------------------------------


def apply_overlay(
    prediction: dict,
    narrative: Narrative,
    catalog: list[Rule],
) -> OverlayResult:
    """Apply the overlay pipeline (P3 + P4) to a prediction dict.

    Args:
        prediction: Dict with pmf_total, expected_fouls, home_expected,
                    away_expected, over_under (from ensemble.predict()).
        narrative:  Parsed Narrative dataclass.
        catalog:    List of Rule objects (from overlay.rules.load_catalog()).

    Returns:
        OverlayResult with all overlay outputs.
    """
    # Step 1: evaluate rules
    fired_rules: list[Rule] = [
        rule for rule in catalog if evaluate_rule(rule, narrative)
    ]

    # Step 2: aggregate effects
    agg = aggregate_effects(fired_rules, narrative.confidence_level)

    # Step 3: capture pre-overlay state
    pre_summary = _pmf_summary(prediction)

    # Step 4: apply PMF tilt (P3)
    tilt_result: TiltResult | None = None
    if fired_rules:
        tilt_result = apply_pmf_tilt(
            prediction=prediction,
            delta_fouls=agg.delta_fouls,
            variance_scale=agg.variance_scale,
        )
        post_prediction = tilt_result.prediction
        suppressed_by_floor_count = 1 if tilt_result.suppressed_by_floor else 0
    else:
        # Identity: copy the prediction as-is
        post_prediction = dict(prediction)
        suppressed_by_floor_count = 0

    post_summary = _pmf_summary(post_prediction)

    # Step 5: build rules_fired list (one entry per fired rule)
    rules_fired_list: list[dict] = []
    for rule in fired_rules:
        rules_fired_list.append(
            {
                "id": rule.id,
                "direction": rule.direction,
                "magnitude_applied": rule.effect.delta_fouls,
                "suppressed_by_floor": (
                    tilt_result.suppressed_by_floor
                    if tilt_result is not None
                    else False
                ),
            }
        )

    # Step 6: kelly_raw / kelly_scaled (P4)
    # These represent the overlay kelly_scale multiplier (1.0 by default).
    # The actual EV/kelly computation happens in run_prediction.py when calling ev.py.
    kelly_raw = 1.0
    kelly_scaled = round(kelly_raw * agg.kelly_scale, 4)

    return OverlayResult(
        pre_pmf_summary=pre_summary,
        post_pmf_summary=post_summary,
        post_prediction=post_prediction,
        rules_fired=rules_fired_list,
        aggregated_effect=agg,
        kelly_raw=kelly_raw,
        kelly_scaled=kelly_scaled,
        suppressed_by_floor_count=suppressed_by_floor_count,
        tilt_result=tilt_result,
    )
