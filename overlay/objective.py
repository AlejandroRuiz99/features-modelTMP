"""
overlay.objective — P1: Objectives injection into state['objectives'].

Public API:
    LABEL_URGENCY_MAP: dict[str, float]
        Maps Narrative label → urgency_base float (matching competitive_context
        _URGENCY_BY_OBJECTIVE values).
    inject_objectives_into_state(state, narrative) -> dict
        Returns a new state dict with the objectives entries patched per narrative.
        Never mutates the input state.
    apply_objective_override: deprecated alias for inject_objectives_into_state
        (will be removed in a future batch).

Implementation note on urgency_base_override (T3.4b)
-----------------------------------------------------
competitive_context._get_objective() reads 'num_categoria' (int) and maps it
via _URGENCY_BY_OBJECTIVE{int → float} to obtain urgency_base.  To avoid
breaking this existing lookup path, inject_objectives_into_state now writes a
SEPARATE key 'urgency_base_override' (float) alongside the original int
'num_categoria' (kept unchanged).

competitive_context._get_objective() has been updated to check
'urgency_base_override' FIRST — if present, it uses the float directly; otherwise
falls back to the existing _URGENCY_BY_OBJECTIVE[int(num_categoria)] path.

Contract:
  - entry['num_categoria']        → original int, always preserved
  - entry['urgency_base_override']→ float in [0.0, 1.0], set by overlay
  - entry['objetivo_label']       → overridden label string
"""

from __future__ import annotations

import copy

from overlay.schema import Narrative

__all__ = [
    "LABEL_URGENCY_MAP",
    "inject_objectives_into_state",
    "apply_objective_override",
]

# ---------------------------------------------------------------------------
# Label → urgency_base mapping
#
# Must mirror competitive_context._URGENCY_BY_OBJECTIVE:
#   num_categoria 1 → 0.85 (titulo)
#   num_categoria 2 → 0.75 (ucl / Champions top4)
#   num_categoria 3 → 0.60 (europa / UEL)
#   num_categoria 5 → 0.25 (mid / media tabla)
#   num_categoria 6 → 0.80 (descenso)
#
# 'salvacion' maps to 0.80 — same relegation-pressure urgency as 'descenso'.
# ---------------------------------------------------------------------------

LABEL_URGENCY_MAP: dict[str, float] = {
    "titulo": 0.85,
    "ucl": 0.75,
    "europa": 0.60,
    "mid": 0.25,
    "salvacion": 0.80,
    "descenso": 0.80,
}


def inject_objectives_into_state(state: dict, narrative: Narrative) -> dict:
    """Patch state['objectives'] with narrative objectives.

    Creates a deep copy of ``state`` and patches the 'objectives' sub-dict.
    The original ``state`` is never modified.

    For each side ('home'/'away') in ``narrative.objectives``:
      - Sets ``objetivo_label`` to the narrative label.
      - Sets ``urgency_base_override`` (float) to:
          - ``urgency_base`` if explicitly provided, else
          - the label-derived value from LABEL_URGENCY_MAP.
      - Preserves ``num_categoria`` at its original value (int from Supabase).

    The separate ``urgency_base_override`` key is read by
    competitive_context._get_objective() in priority over the int-keyed lookup.

    Args:
        state: The current prediction state dict (contains 'objectives').
        narrative: Parsed Narrative dataclass (MUST contain objectives).

    Returns:
        A deep copy of state with patched objectives (original unmodified).
    """
    new_state: dict = copy.deepcopy(state)

    objectives: dict[str, dict] = new_state.setdefault("objectives", {})

    team_for_side: dict[str, str] = {
        "home": narrative.match.home,
        "away": narrative.match.away,
    }

    for side, obj in narrative.objectives.items():
        team_name = team_for_side.get(side)
        if team_name is None:
            continue

        urgency_base = (
            obj.urgency_base
            if obj.urgency_base is not None
            else LABEL_URGENCY_MAP.get(obj.label, 0.25)
        )

        entry = objectives.setdefault(team_name, {})
        entry["objetivo_label"] = obj.label
        # Write urgency as a separate key — num_categoria (int) is preserved unchanged
        entry["urgency_base_override"] = urgency_base

    return new_state


# Deprecated alias — will be removed in a future batch
apply_objective_override = inject_objectives_into_state
