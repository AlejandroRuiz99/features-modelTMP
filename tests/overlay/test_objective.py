"""
tests/overlay/test_objective.py — Strict TDD for overlay.objective.

Tests (10 cases):
  1. Label-only override patches urgency_base_override with label-derived urgency_base
  2. Label + urgency_base uses explicit float value in urgency_base_override
  3. Only-home override leaves away state unchanged
  4. Only-away override leaves home state unchanged
  5. Both sides override patches both correctly
  6. Supabase rows NOT mutated — deep copy verified (original state is unmodified)
  7. num_categoria (int) is preserved as-is when override is applied
  8. urgency_base_override key is set alongside original num_categoria
  9. Deprecated alias apply_objective_override is callable and produces same result
  10. inject_objectives_into_state name is exported in __all__
"""

from __future__ import annotations

import copy
import warnings

from overlay.objective import LABEL_URGENCY_MAP, inject_objectives_into_state
from overlay.schema import Narrative, NarrativeMatch, ObjectiveOverride

# Backward-compat alias
from overlay.objective import apply_objective_override

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOME_TEAM = "Espanyol"
_AWAY_TEAM = "Levante"


def _make_state(
    home_label: str = "mid",
    home_cat: int = 5,
    away_label: str = "mid",
    away_cat: int = 5,
) -> dict:
    """Minimal state dict with objectives for home and away teams."""
    return {
        "objectives": {
            _HOME_TEAM: {
                "objetivo_label": home_label,
                "num_categoria": home_cat,
                "competiciones_activas": False,
            },
            _AWAY_TEAM: {
                "objetivo_label": away_label,
                "num_categoria": away_cat,
                "competiciones_activas": False,
            },
        }
    }


def _make_narrative(
    objectives: dict | None = None,
    confidence_level: int = 4,
) -> Narrative:
    """Build a Narrative with the given objectives.

    Since objectives is REQUIRED (both home and away), partial inputs
    are auto-filled with default mid/mid objectives.
    """
    default_obj = {
        "home": ObjectiveOverride(label="mid"),
        "away": ObjectiveOverride(label="mid"),
    }
    if objectives is not None:
        for side, vals in objectives.items():
            default_obj[side] = ObjectiveOverride(**vals)
    return Narrative(
        match=NarrativeMatch(home=_HOME_TEAM, away=_AWAY_TEAM, date="2026-04-27"),
        confidence_level=confidence_level,
        objectives=default_obj,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInjectObjectivesIntoState:
    def test_label_only_override_patches_urgency_with_label_derived_value(self) -> None:
        """Label-only override → urgency_base_override set from LABEL_URGENCY_MAP."""
        state = _make_state()
        narr = _make_narrative(objectives={"home": {"label": "descenso"}})
        new_state = inject_objectives_into_state(state, narr)

        home_entry = new_state["objectives"][_HOME_TEAM]
        assert home_entry["objetivo_label"] == "descenso"
        # urgency_base_override set from LABEL_URGENCY_MAP["descenso"]
        expected_urgency = LABEL_URGENCY_MAP["descenso"]
        assert "urgency_base_override" in home_entry, (
            "Expected 'urgency_base_override' key in patched entry"
        )
        assert abs(home_entry["urgency_base_override"] - expected_urgency) < 1e-9, (
            f"Expected urgency_base_override = {expected_urgency}, "
            f"got {home_entry['urgency_base_override']}"
        )

    def test_label_plus_urgency_base_uses_explicit_float(self) -> None:
        """label + urgency_base=0.71 → explicit value in urgency_base_override."""
        state = _make_state()
        narr = _make_narrative(
            objectives={"home": {"label": "salvacion", "urgency_base": 0.71}}
        )
        new_state = inject_objectives_into_state(state, narr)

        home_entry = new_state["objectives"][_HOME_TEAM]
        assert home_entry["objetivo_label"] == "salvacion"
        assert "urgency_base_override" in home_entry, (
            "Expected 'urgency_base_override' key in patched entry"
        )
        assert abs(home_entry["urgency_base_override"] - 0.71) < 1e-9, (
            f"Expected urgency_base_override = 0.71, got {home_entry['urgency_base_override']}"
        )

    def test_num_categoria_preserved_as_int_when_override_applied(self) -> None:
        """num_categoria (original int) stays unchanged when override is applied."""
        original_home_cat = 5  # mid
        state = _make_state(home_cat=original_home_cat)
        narr = _make_narrative(
            objectives={"home": {"label": "descenso", "urgency_base": 0.80}}
        )
        new_state = inject_objectives_into_state(state, narr)

        home_entry = new_state["objectives"][_HOME_TEAM]
        # num_categoria keeps original int value, NOT overwritten by float urgency
        assert home_entry["num_categoria"] == original_home_cat, (
            f"Expected num_categoria to remain {original_home_cat}, "
            f"got {home_entry['num_categoria']}"
        )
        # urgency_base_override is the float
        assert abs(home_entry["urgency_base_override"] - 0.80) < 1e-9

    def test_only_home_override_leaves_away_at_default(self) -> None:
        """When only home objective differs, away gets the default label from narrative."""
        state = _make_state(away_label="europa", away_cat=3)
        narr = _make_narrative(objectives={"home": {"label": "descenso"}})
        new_state = inject_objectives_into_state(state, narr)

        away_entry = new_state["objectives"][_AWAY_TEAM]
        # Away gets the narrative's away objective (default mid)
        assert away_entry["objetivo_label"] == "mid"
        assert away_entry["num_categoria"] == 3

    def test_only_away_override_leaves_home_at_default(self) -> None:
        """When only away objective differs, home gets the default label from narrative."""
        state = _make_state(home_label="ucl", home_cat=2)
        narr = _make_narrative(objectives={"away": {"label": "descenso"}})
        new_state = inject_objectives_into_state(state, narr)

        home_entry = new_state["objectives"][_HOME_TEAM]
        # Home gets the narrative's home objective (default mid)
        assert home_entry["objetivo_label"] == "mid"
        assert home_entry["num_categoria"] == 2

    def test_both_sides_override_patches_both(self) -> None:
        """Both home and away objectives are patched when both provided."""
        state = _make_state()
        narr = _make_narrative(
            objectives={
                "home": {"label": "salvacion", "urgency_base": 0.70},
                "away": {"label": "descenso", "urgency_base": 0.85},
            }
        )
        new_state = inject_objectives_into_state(state, narr)

        home_entry = new_state["objectives"][_HOME_TEAM]
        away_entry = new_state["objectives"][_AWAY_TEAM]
        assert home_entry["objetivo_label"] == "salvacion"
        assert abs(home_entry["urgency_base_override"] - 0.70) < 1e-9
        assert away_entry["objetivo_label"] == "descenso"
        assert abs(away_entry["urgency_base_override"] - 0.85) < 1e-9

    def test_original_state_not_mutated(self) -> None:
        """inject_objectives_into_state returns a deep copy — original state is unmodified."""
        state = _make_state(home_label="mid", home_cat=5)
        original_state = copy.deepcopy(state)
        narr = _make_narrative(objectives={"home": {"label": "descenso"}})
        new_state = inject_objectives_into_state(state, narr)

        # Original state must be unmodified
        assert state == original_state, (
            "inject_objectives_into_state mutated the input state dict"
        )
        # New state must differ (home was patched)
        assert new_state["objectives"][_HOME_TEAM]["objetivo_label"] == "descenso"
        assert state["objectives"][_HOME_TEAM]["objetivo_label"] == "mid"


class TestDeprecatedAlias:
    def test_apply_objective_override_is_callable(self) -> None:
        """Deprecated alias apply_objective_override produces same result as inject_objectives_into_state."""
        state = _make_state()
        narr = _make_narrative(
            objectives={"home": {"label": "descenso", "urgency_base": 0.80}}
        )
        result_new = inject_objectives_into_state(state, narr)
        result_old = apply_objective_override(state, narr)
        assert result_new == result_old

    def test_inject_objectives_into_state_in_all(self) -> None:
        """inject_objectives_into_state is exported in __all__."""
        from overlay.objective import __all__

        assert "inject_objectives_into_state" in __all__
        assert "apply_objective_override" in __all__
