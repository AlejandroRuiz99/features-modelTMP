"""
parsers.narrative_parser — Narrative YAML validator and cache-check.

This module does NOT call any LLM. It provides:
  - NARRATIVE_PROMPT_TEMPLATE: prompt string the SKILL.md LLM agent uses
  - validate_narrative_yaml(): delegates to overlay/loader.py for validation
  - narrative_cached(): checks if a parsed YAML already exists

The actual free-text → YAML parsing is done by the LLM agent in the skill.
"""

from __future__ import annotations

from pathlib import Path

from overlay.loader import load_narrative
from overlay.schema import Narrative

__all__ = [
    "NARRATIVE_PROMPT_TEMPLATE",
    "narrative_cached",
    "validate_narrative_yaml",
]

# ---------------------------------------------------------------------------
# LLM prompt template
# ---------------------------------------------------------------------------

NARRATIVE_PROMPT_TEMPLATE: str = """\
Parse the following match context into YAML format.
REQUIRED fields: match (home, away, date, jornada), confidence_level.

Match: {home} vs {away}, {date}, J{jornada}

User text:
---
{raw_text}
---

Output YAML following this exact schema:
match:
  home: "{home}"
  away: "{away}"
  date: "{date}"
  jornada: {jornada}

objectives:
  home:
    label: <one of: descenso, salvacion, mid, europa, ucl, titulo>
    urgency_base: <float 0.0-1.0, optional>
  away:
    label: <same options>
    urgency_base: <float 0.0-1.0, optional>

stakes:
  home: <0-5>
  away: <0-5>
  notes: "<brief>"

intensity_override: <0-5 or null>
physicality_bias: <-2 to +2 or null>
referee_factor: <-2 to +2 or null>
special_flags: [<from valid list>]
confidence_level: <1-5>
notes: "<brief>"

Valid special_flags: stakes_both_relegation, stakes_one_relegation, derbi,
copa_recent_extra_time, european_midweek, coach_debut, coach_pressure,
last_round_drama, weather_extreme, b_team_expected, key_injuries_home,
key_injuries_away, morbo, dead_rubber, late_season, early_season,
physical_clash, strict_ref_announced, permissive_ref_announced

IMPORTANT: confidence_level is REQUIRED (integer 1-5).
Output ONLY the YAML block. No markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# Validation (delegates to overlay/loader.py)
# ---------------------------------------------------------------------------


def validate_narrative_yaml(path: Path) -> Narrative:
    """Load and validate a narrative YAML file.

    Delegates to overlay.loader.load_narrative for full schema validation.

    Args:
        path: Path to the narrative YAML file.

    Returns:
        Validated Narrative dataclass.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is malformed or fails schema validation.
    """
    return load_narrative(path)


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------


def narrative_cached(slug: str, narratives_dir: Path) -> Path | None:
    """Check if a parsed narrative YAML already exists (run resume cache).

    Args:
        slug: Match slug (e.g. "realmadrid_vs_mallorca_2026-05-04").
        narratives_dir: Directory where narrative YAMLs are stored.

    Returns:
        Path to the existing YAML file if it exists and is non-empty,
        else None.
    """
    if not narratives_dir.exists():
        return None

    narr_path = narratives_dir / f"{slug}.yaml"
    if not narr_path.exists():
        return None

    # Non-empty check
    try:
        content = narr_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not content:
        return None

    return narr_path
