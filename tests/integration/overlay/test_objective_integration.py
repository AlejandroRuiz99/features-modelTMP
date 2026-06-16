"""
tests/integration/overlay/test_objective_integration.py — Integration tests for P1.

Verifies that apply_objectives produces a patched state that competitive_context
reads with the overridden urgency values.

Tests (3 cases):
  1. With override home label "descenso" urgency=0.80 → objetivo_label and
     urgency_base_override are correct in the patched state.
  2. Original state dict is unchanged after apply_objectives.
  3. competitive_context._get_objective reads urgency_base_override directly
     (float) instead of going through the int-category lookup.

Strategy: use a synthetic minimal state dict (no Supabase) and directly call
competitive_context._get_objective() to verify the override is picked up correctly.
This avoids the need for network access while testing the actual integration point.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

# Ensure features_generator is on sys.path so transformation can be imported
ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_DIR = ROOT / "features_generator"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from overlay.objective import inject_objectives_into_state  # noqa: E402
from overlay.schema import Narrative, NarrativeMatch, ObjectiveOverride  # noqa: E402
from transformation.competitive_context import _get_objective  # noqa: E402


_HOME_TEAM = "Espanol"
_AWAY_TEAM = "Levante"


def _make_fixture_state(
    home_label: str = "mid",
    home_cat: int = 5,
    away_label: str = "mid",
    away_cat: int = 5,
) -> dict:
    """Minimal synthetic state with objectives for home + away teams."""
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
        },
        "scores": {},
        "cal_index": {},
    }


def _make_narrative_with_home_override(
    label: str = "descenso",
    urgency_base: float | None = None,
) -> Narrative:
    oo_kwargs: dict = {"label": label}
    if urgency_base is not None:
        oo_kwargs["urgency_base"] = urgency_base
    return Narrative(
        match=NarrativeMatch(home=_HOME_TEAM, away=_AWAY_TEAM, date="2026-04-27"),
        confidence_level=4,
        objectives={
            "home": ObjectiveOverride(**oo_kwargs),
            "away": ObjectiveOverride(label="mid"),
        },
    )


class TestObjectiveIntegration:
    def test_patched_state_has_overridden_objetivo_label_and_urgency(self) -> None:
        """After apply_objectives, objectives[home_team] has
        'objetivo_label' == 'descenso' and 'urgency_base_override' == 0.80."""
        state = _make_fixture_state(home_label="mid", home_cat=5)
        narr = _make_narrative_with_home_override(label="descenso", urgency_base=0.80)

        new_state = inject_objectives_into_state(state, narr)

        home_obj = new_state["objectives"][_HOME_TEAM]
        assert home_obj["objetivo_label"] == "descenso", (
            f"Expected 'descenso', got '{home_obj['objetivo_label']}'"
        )
        # urgency stored in urgency_base_override (NOT in num_categoria)
        assert "urgency_base_override" in home_obj, (
            "Expected 'urgency_base_override' key in patched entry"
        )
        assert abs(home_obj["urgency_base_override"] - 0.80) < 1e-9, (
            f"Expected urgency_base_override=0.80, got {home_obj['urgency_base_override']}"
        )
        # num_categoria stays as original int (5 = mid)
        assert home_obj["num_categoria"] == 5, (
            f"num_categoria should be original int 5, got {home_obj['num_categoria']}"
        )
        # Away should be unchanged (still 'mid', cat=5)
        away_obj = new_state["objectives"][_AWAY_TEAM]
        assert away_obj["objetivo_label"] == "mid"
        assert away_obj["num_categoria"] == 5

    def test_original_state_unchanged_after_override(self) -> None:
        """apply_objectives does not mutate the input state."""
        state = _make_fixture_state(home_label="mid", home_cat=5)
        state_before = copy.deepcopy(state)
        narr = _make_narrative_with_home_override(label="descenso", urgency_base=0.80)

        _new_state = inject_objectives_into_state(state, narr)

        # Original state must be bit-for-bit identical to what it was before
        assert state == state_before, "apply_objectives mutated the original state dict"

    def test_get_objective_reads_urgency_base_override_directly(self) -> None:
        """competitive_context._get_objective uses urgency_base_override float
        directly when present, bypassing the int-category lookup."""
        state = _make_fixture_state(home_label="mid", home_cat=5)
        narr = _make_narrative_with_home_override(label="descenso", urgency_base=0.80)

        new_state = inject_objectives_into_state(state, narr)

        objectives = new_state["objectives"]
        label, urgency, comp_activas = _get_objective(_HOME_TEAM, objectives)

        assert label == "descenso", f"Expected label='descenso', got {label!r}"
        assert urgency is not None, "urgency should not be None"
        assert abs(urgency - 0.80) < 1e-9, (
            f"Expected urgency=0.80 from urgency_base_override, got {urgency}"
        )
        assert comp_activas is False
