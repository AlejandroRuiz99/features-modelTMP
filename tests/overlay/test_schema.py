"""
tests/overlay/test_schema.py — Unit tests for overlay.schema dataclasses.

Tests (11 cases — Strict TDD):
  1. Valid full narrative passes with no errors
  2. Missing confidence_level raises ValueError
  3. Invalid objectives.home.label raises ValueError
  4. Unknown special_flags entry raises ValueError
  5. confidence_level out of range (6) raises ValueError
  6. physicality_bias out of range (+3) raises ValueError
  7. All optional fields absent -> defaults applied correctly
  8. Unknown top-level field raises ValueError
  9. Narrative without objectives raises TypeError (REQUIRED)
  10. Narrative with only home objectives raises ValueError
  11. Narrative with only away objectives raises ValueError
"""

from __future__ import annotations

import pytest

from overlay.schema import (
    Narrative,
    NarrativeMatch,
    NarrativeStakes,
    ObjectiveOverride,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_match(**kwargs) -> NarrativeMatch:
    defaults = {"home": "Espanyol", "away": "Levante", "date": "2026-04-27"}
    return NarrativeMatch(**{**defaults, **kwargs})


def _make_minimal_narrative(**kwargs) -> Narrative:
    """Minimal valid narrative (required fields: confidence_level, objectives)."""
    defaults = {
        "match": _make_match(),
        "confidence_level": 3,
        "objectives": {
            "home": ObjectiveOverride(label="mid"),
            "away": ObjectiveOverride(label="mid"),
        },
    }
    return Narrative(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Case 1: Valid full narrative passes
# ---------------------------------------------------------------------------


class TestValidFullNarrative:
    def test_full_narrative_no_error(self) -> None:
        """A complete narrative with all optional fields filled passes validation."""
        narr = Narrative(
            match=NarrativeMatch(
                home="Espanyol",
                away="Levante",
                date="2026-04-27",
                competition="La Liga",
                jornada=32,
            ),
            confidence_level=4,
            objectives={
                "home": ObjectiveOverride(label="salvacion", urgency_base=0.65),
                "away": ObjectiveOverride(label="descenso", urgency_base=0.80),
            },
            stakes=NarrativeStakes(home=4, away=5, notes="Relegation battle"),
            rotations={"home": 0, "away": 0},
            intensity_override=4,
            physicality_bias=1,
            referee_factor=0,
            special_flags=["stakes_both_relegation", "late_season", "physical_clash"],
            notes="Full narrative test",
        )
        assert narr.confidence_level == 4
        assert narr.match.home == "Espanyol"
        assert narr.special_flags == [
            "stakes_both_relegation",
            "late_season",
            "physical_clash",
        ]


# ---------------------------------------------------------------------------
# Case 2: Missing confidence_level raises ValueError
# ---------------------------------------------------------------------------


class TestMissingRequiredField:
    def test_missing_confidence_level_raises(self) -> None:
        """Narrative without confidence_level must fail with clear error."""
        with pytest.raises((ValueError, TypeError)):
            # confidence_level has no default → TypeError from dataclass
            Narrative(match=_make_match())  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Case 3: Invalid objective_override.home.label raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidObjectiveLabel:
    def test_invalid_label_raises(self) -> None:
        """label='campeon' is not in VALID_LABELS → must raise ValueError."""
        with pytest.raises(ValueError, match="label"):
            ObjectiveOverride(label="campeon")


# ---------------------------------------------------------------------------
# Case 4: Unknown special_flags entry raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownSpecialFlag:
    def test_unknown_flag_raises(self) -> None:
        """Flag 'foo_flag' not in VALID_FLAGS → must raise ValueError."""
        with pytest.raises(ValueError, match=r"special_flags|flag"):
            Narrative(
                match=_make_match(),
                confidence_level=3,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
                special_flags=["derbi", "foo_flag"],
            )


# ---------------------------------------------------------------------------
# Case 5: confidence_level out of range (6) raises ValueError
# ---------------------------------------------------------------------------


class TestConfidenceLevelOutOfRange:
    def test_confidence_level_above_max_raises(self) -> None:
        """confidence_level=6 exceeds max of 5 → must raise ValueError."""
        with pytest.raises(ValueError, match="confidence_level"):
            Narrative(
                match=_make_match(),
                confidence_level=6,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
            )

    def test_confidence_level_below_min_raises(self) -> None:
        """confidence_level=0 is below min of 1 → must raise ValueError."""
        with pytest.raises(ValueError, match="confidence_level"):
            Narrative(
                match=_make_match(),
                confidence_level=0,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
            )


# ---------------------------------------------------------------------------
# Case 6: physicality_bias out of range (+3) raises ValueError
# ---------------------------------------------------------------------------


class TestPhysicalityBiasOutOfRange:
    def test_physicality_bias_too_high_raises(self) -> None:
        """physicality_bias=3 exceeds max of 2 → must raise ValueError."""
        with pytest.raises(ValueError, match="physicality_bias"):
            Narrative(
                match=_make_match(),
                confidence_level=3,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
                physicality_bias=3,
            )

    def test_physicality_bias_too_low_raises(self) -> None:
        """physicality_bias=-3 below min of -2 → must raise ValueError."""
        with pytest.raises(ValueError, match="physicality_bias"):
            Narrative(
                match=_make_match(),
                confidence_level=3,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
                physicality_bias=-3,
            )


# ---------------------------------------------------------------------------
# Case 7: All optional fields absent → defaults applied
# ---------------------------------------------------------------------------


class TestOptionalFieldDefaults:
    def test_all_optional_absent_uses_defaults(self) -> None:
        """Narrative with only required fields → optionals default to None/[]."""
        narr = _make_minimal_narrative()
        assert narr.objectives is not None
        assert narr.objectives["home"].label == "mid"
        assert narr.stakes is None
        assert narr.rotations is None
        assert narr.intensity_override is None
        assert narr.physicality_bias is None
        assert narr.referee_factor is None
        assert narr.special_flags == []
        assert narr.notes is None


# ---------------------------------------------------------------------------
# Case 8: Unknown top-level field raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownTopLevelField:
    def test_unknown_field_raises(self) -> None:
        """A field not in the schema (foo_bar) must cause a clear error."""
        with pytest.raises((ValueError, TypeError)):
            Narrative(  # type: ignore[call-arg]
                match=_make_match(),
                confidence_level=3,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
                foo_bar=99,
            )


# ---------------------------------------------------------------------------
# Case 9: Narrative without objectives raises TypeError (REQUIRED)
# ---------------------------------------------------------------------------


class TestMissingObjectives:
    def test_no_objectives_raises_type_error(self) -> None:
        """Narrative without objectives must fail — objectives is REQUIRED."""
        with pytest.raises(TypeError):
            Narrative(  # type: ignore[call-arg]
                match=_make_match(),
                confidence_level=3,
            )


# ---------------------------------------------------------------------------
# Case 10: Narrative with only home objectives raises ValueError
# ---------------------------------------------------------------------------


class TestOnlyHomeObjectives:
    def test_only_home_objectives_raises(self) -> None:
        """objectives with only 'home' key must fail — 'away' is required."""
        with pytest.raises(ValueError, match="away"):
            Narrative(
                match=_make_match(),
                confidence_level=3,
                objectives={"home": ObjectiveOverride(label="mid")},
            )


# ---------------------------------------------------------------------------
# Case 11: Narrative with only away objectives raises ValueError
# ---------------------------------------------------------------------------


class TestOnlyAwayObjectives:
    def test_only_away_objectives_raises(self) -> None:
        """objectives with only 'away' key must fail — 'home' is required."""
        with pytest.raises(ValueError, match="home"):
            Narrative(
                match=_make_match(),
                confidence_level=3,
                objectives={"away": ObjectiveOverride(label="mid")},
            )
