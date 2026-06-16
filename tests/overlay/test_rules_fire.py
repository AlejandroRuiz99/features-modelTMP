"""
tests/overlay/test_rules_fire.py — Parametrized tests for each of the 15 rules.

For each rule: one narrative that FIRES the rule, one that does NOT.
30 total parametrized cases.

Tests load rules from the actual overlay/rules.yaml catalog.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from overlay.rules import evaluate_rule, load_catalog
from overlay.schema import Narrative, NarrativeMatch, NarrativeStakes, ObjectiveOverride

RULES_YAML = Path(__file__).parent.parent.parent / "overlay" / "rules.yaml"


# ---------------------------------------------------------------------------
# Load catalog once at module level
# ---------------------------------------------------------------------------

_ALL_RULES = {r.id: r for r in load_catalog(RULES_YAML)}


def _rule(rule_id: str):
    assert rule_id in _ALL_RULES, (
        f"Rule '{rule_id}' not found in catalog. Available: {list(_ALL_RULES)}"
    )
    return _ALL_RULES[rule_id]


# ---------------------------------------------------------------------------
# Narrative factories for each rule
# ---------------------------------------------------------------------------


def _match() -> NarrativeMatch:
    return NarrativeMatch(home="TeamA", away="TeamB", date="2026-01-01")


def _narrative_fires_both_relegation() -> Narrative:
    """both_relegation_up fires: both teams in descenso/salvacion."""
    return Narrative(
        match=_match(),
        confidence_level=4,
        objectives={
            "home": ObjectiveOverride(label="salvacion"),
            "away": ObjectiveOverride(label="descenso"),
        },
    )


def _narrative_nofires_both_relegation() -> Narrative:
    """both_relegation_up does NOT fire: only home team in relegation."""
    return Narrative(
        match=_match(),
        confidence_level=4,
        objectives={
            "home": ObjectiveOverride(label="salvacion"),
            "away": ObjectiveOverride(label="europa"),
        },
    )


def _narrative_fires_one_relegation_high_stakes() -> Narrative:
    """one_relegation_high_stakes_up fires: home in salvacion, away stakes>=3."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        objectives={
            "home": ObjectiveOverride(label="salvacion"),
            "away": ObjectiveOverride(label="mid"),
        },
        stakes=NarrativeStakes(home=2, away=4),
    )


def _narrative_nofires_one_relegation_high_stakes() -> Narrative:
    """one_relegation_high_stakes_up does NOT fire: both mid, low stakes."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="europa"),
        },
        stakes=NarrativeStakes(home=1, away=1),
    )


def _narrative_fires_derbi() -> Narrative:
    """derbi_intensity_up fires: derbi flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["derbi"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_derbi() -> Narrative:
    """derbi_intensity_up does NOT fire: no derbi flag."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_physical_clash() -> Narrative:
    """physical_clash_up fires: physical_clash flag + physicality_bias=1."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["physical_clash"],
        physicality_bias=1,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_physical_clash() -> Narrative:
    """physical_clash_up does NOT fire: no physical_clash flag."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        physicality_bias=2,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_coach_pressure() -> Narrative:
    """coach_pressure_up fires: coach_pressure flag + intensity_override=4."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["coach_pressure"],
        intensity_override=4,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_coach_pressure() -> Narrative:
    """coach_pressure_up does NOT fire: no coach_pressure flag."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        intensity_override=5,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_last_round_drama() -> Narrative:
    """last_round_drama_up fires: last_round_drama flag + high stakes both."""
    return Narrative(
        match=_match(),
        confidence_level=4,
        special_flags=["last_round_drama"],
        stakes=NarrativeStakes(home=5, away=4),
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_last_round_drama() -> Narrative:
    """last_round_drama_up does NOT fire: stakes too low."""
    return Narrative(
        match=_match(),
        confidence_level=4,
        special_flags=["last_round_drama"],
        stakes=NarrativeStakes(home=3, away=2),
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_strict_ref_physical() -> Narrative:
    """strict_ref_physical_up fires: strict_ref_announced + physicality_bias=1."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["strict_ref_announced"],
        physicality_bias=1,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_strict_ref_physical() -> Narrative:
    """strict_ref_physical_up does NOT fire: permissive ref instead."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["permissive_ref_announced"],
        physicality_bias=1,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_high_intensity_override() -> Narrative:
    """high_intensity_override_up fires: intensity>=4 + physicality>=1."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        intensity_override=4,
        physicality_bias=1,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_high_intensity_override() -> Narrative:
    """high_intensity_override_up does NOT fire: intensity too low."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        intensity_override=3,
        physicality_bias=2,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_dead_rubber() -> Narrative:
    """dead_rubber_down fires: dead_rubber flag + stakes<=1 both."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["dead_rubber"],
        stakes=NarrativeStakes(home=0, away=1),
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_dead_rubber() -> Narrative:
    """dead_rubber_down does NOT fire: stakes too high."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["dead_rubber"],
        stakes=NarrativeStakes(home=3, away=2),
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_heavy_rotations_both() -> Narrative:
    """heavy_rotations_both_down fires: rotations>=3 for both teams."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        rotations={"home": 4, "away": 3},
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_heavy_rotations_both() -> Narrative:
    """heavy_rotations_both_down does NOT fire: only one team rotates."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        rotations={"home": 4, "away": 1},
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_one_team_rotation() -> Narrative:
    """one_team_rotation_down fires: at least one team rotations>=4."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        rotations={"home": 4, "away": 1},
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_one_team_rotation() -> Narrative:
    """one_team_rotation_down does NOT fire: no team rotates >=4."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        rotations={"home": 3, "away": 3},
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_permissive_ref_technical() -> Narrative:
    """permissive_ref_technical_down fires: permissive_ref + physicality<=0."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["permissive_ref_announced"],
        physicality_bias=0,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_permissive_ref_technical() -> Narrative:
    """permissive_ref_technical_down does NOT fire: physicality too high."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["permissive_ref_announced"],
        physicality_bias=1,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_b_team_expected() -> Narrative:
    """b_team_expected_down fires: b_team_expected flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["b_team_expected"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_b_team_expected() -> Narrative:
    """b_team_expected_down does NOT fire: flag absent."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_european_fatigue() -> Narrative:
    """european_fatigue_down fires: european_midweek + rotations<=1 both."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["european_midweek"],
        rotations={"home": 1, "away": 0},
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_european_fatigue() -> Narrative:
    """european_fatigue_down does NOT fire: heavy rotations (>1)."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["european_midweek"],
        rotations={"home": 3, "away": 2},
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_copa_extra_time() -> Narrative:
    """copa_extra_time_fatigue_down fires: copa_recent_extra_time flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["copa_recent_extra_time"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_copa_extra_time() -> Narrative:
    """copa_extra_time_fatigue_down does NOT fire: flag absent."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_coach_pressure_volatile() -> Narrative:
    """coach_pressure_volatile fires: coach_pressure flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["coach_pressure"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_coach_pressure_volatile() -> Narrative:
    """coach_pressure_volatile does NOT fire: neither coach_pressure nor coach_debut flag."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["late_season"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_coach_debut_volatile() -> Narrative:
    """coach_pressure_volatile also fires: coach_debut flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["coach_debut"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_key_injuries_volatile() -> Narrative:
    """key_injuries_disrupted_volatile fires: key_injuries_home flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["key_injuries_home"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_fires_key_injuries_away_volatile() -> Narrative:
    """key_injuries_disrupted_volatile also fires: key_injuries_away flag present."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["key_injuries_away"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


def _narrative_nofires_key_injuries_volatile() -> Narrative:
    """key_injuries_disrupted_volatile does NOT fire: no injury-related flags."""
    return Narrative(
        match=_match(),
        confidence_level=3,
        special_flags=["derbi"],
        objectives={
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    )


# ---------------------------------------------------------------------------
# Parametrized test cases: (rule_id, fires_narrative, nofires_narrative)
# ---------------------------------------------------------------------------

RULE_FIRE_CASES = [
    (
        "both_relegation_up",
        _narrative_fires_both_relegation,
        _narrative_nofires_both_relegation,
    ),
    (
        "one_relegation_high_stakes_up",
        _narrative_fires_one_relegation_high_stakes,
        _narrative_nofires_one_relegation_high_stakes,
    ),
    ("derbi_intensity_up", _narrative_fires_derbi, _narrative_nofires_derbi),
    (
        "physical_clash_up",
        _narrative_fires_physical_clash,
        _narrative_nofires_physical_clash,
    ),
    (
        "coach_pressure_up",
        _narrative_fires_coach_pressure,
        _narrative_nofires_coach_pressure,
    ),
    (
        "last_round_drama_up",
        _narrative_fires_last_round_drama,
        _narrative_nofires_last_round_drama,
    ),
    (
        "strict_ref_physical_up",
        _narrative_fires_strict_ref_physical,
        _narrative_nofires_strict_ref_physical,
    ),
    (
        "high_intensity_override_up",
        _narrative_fires_high_intensity_override,
        _narrative_nofires_high_intensity_override,
    ),
    ("dead_rubber_down", _narrative_fires_dead_rubber, _narrative_nofires_dead_rubber),
    (
        "heavy_rotations_both_down",
        _narrative_fires_heavy_rotations_both,
        _narrative_nofires_heavy_rotations_both,
    ),
    (
        "one_team_rotation_down",
        _narrative_fires_one_team_rotation,
        _narrative_nofires_one_team_rotation,
    ),
    (
        "permissive_ref_technical_down",
        _narrative_fires_permissive_ref_technical,
        _narrative_nofires_permissive_ref_technical,
    ),
    (
        "b_team_expected_down",
        _narrative_fires_b_team_expected,
        _narrative_nofires_b_team_expected,
    ),
    (
        "european_fatigue_down",
        _narrative_fires_european_fatigue,
        _narrative_nofires_european_fatigue,
    ),
    (
        "copa_extra_time_fatigue_down",
        _narrative_fires_copa_extra_time,
        _narrative_nofires_copa_extra_time,
    ),
    # Volatility rules
    (
        "coach_pressure_volatile",
        _narrative_fires_coach_pressure_volatile,
        _narrative_nofires_coach_pressure_volatile,
    ),
    (
        "key_injuries_disrupted_volatile",
        _narrative_fires_key_injuries_volatile,
        _narrative_nofires_key_injuries_volatile,
    ),
]


class TestRulesFire:
    @pytest.mark.parametrize("rule_id,fires_fn,nofires_fn", RULE_FIRE_CASES)
    def test_rule_fires_on_matching_narrative(
        self, rule_id: str, fires_fn, nofires_fn
    ) -> None:
        """Rule fires when narrative satisfies trigger conditions."""
        rule = _rule(rule_id)
        narr = fires_fn()
        assert evaluate_rule(rule, narr) is True, (
            f"Rule '{rule_id}' should fire for narrative: "
            f"flags={narr.special_flags}, objective={narr.objectives}, "
            f"stakes={narr.stakes}, rotations={narr.rotations}, "
            f"physicality_bias={narr.physicality_bias}, intensity={narr.intensity_override}"
        )

    @pytest.mark.parametrize("rule_id,fires_fn,nofires_fn", RULE_FIRE_CASES)
    def test_rule_does_not_fire_on_counter_narrative(
        self, rule_id: str, fires_fn, nofires_fn
    ) -> None:
        """Rule does NOT fire when narrative does not satisfy trigger conditions."""
        rule = _rule(rule_id)
        narr = nofires_fn()
        assert evaluate_rule(rule, narr) is False, (
            f"Rule '{rule_id}' should NOT fire for counter-narrative: "
            f"flags={narr.special_flags}, objective={narr.objectives}, "
            f"stakes={narr.stakes}, rotations={narr.rotations}, "
            f"physicality_bias={narr.physicality_bias}, intensity={narr.intensity_override}"
        )
