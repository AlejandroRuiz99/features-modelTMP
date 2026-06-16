"""
overlay.loader — YAML file discovery, parsing, and validation.

Parses narrative YAML files into Narrative dataclasses.
Fails fast on missing files, parse errors, or schema violations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from overlay.schema import (
    Narrative,
    NarrativeMatch,
    NarrativeStakes,
    ObjectiveOverride,
)

__all__ = ["discover_narratives", "load_narrative"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_narrative(path: Path | str) -> Narrative:
    """Load and validate a narrative YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated Narrative dataclass.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is malformed or fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Narrative file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error in {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Narrative file {path.name} must be a YAML mapping, "
            f"got {type(data).__name__}"
        )

    return _build_narrative(data, source_name=path.name)


def discover_narratives(directory: Path | str) -> dict[str, Narrative]:
    """Discover and load all YAML narrative files in a directory.

    Args:
        directory: Directory to scan for *.yaml files.

    Returns:
        Mapping from filename stem → Narrative. Files that fail to parse
        are skipped with a warning logged to stderr.
    """
    import sys

    directory = Path(directory)
    result: dict[str, Narrative] = {}

    for yaml_path in sorted(directory.glob("*.yaml")):
        try:
            narr = load_narrative(yaml_path)
            result[yaml_path.stem] = narr
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"[overlay.loader] WARNING: Skipping {yaml_path.name}: {exc}",
                file=sys.stderr,
            )

    return result


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_narrative(data: dict[str, Any], source_name: str = "<unknown>") -> Narrative:
    """Construct a Narrative dataclass from a raw YAML dict.

    Raises:
        ValueError: On unknown fields or missing required fields.
    """
    _check_unknown_fields(
        data,
        allowed={
            "match",
            "confidence_level",
            "objectives",
            "stakes",
            "rotations",
            "intensity_override",
            "physicality_bias",
            "referee_factor",
            "special_flags",
            "notes",
        },
        context=source_name,
    )

    # Required: match
    if "match" not in data:
        raise ValueError(f"{source_name}: missing required field 'match'")
    match = _build_match(data["match"], source_name=source_name)

    # Required: confidence_level
    if "confidence_level" not in data:
        raise ValueError(f"{source_name}: missing required field 'confidence_level'")

    # Required: objectives (both home and away)
    if "objectives" not in data or data["objectives"] is None:
        raise ValueError(
            f"{source_name}: missing required field 'objectives'. "
            f"Narratives must specify objectives for both home and away teams."
        )
    raw_obj = data["objectives"]
    if not isinstance(raw_obj, dict):
        raise ValueError(
            f"{source_name}: 'objectives' must be a mapping, "
            f"got {type(raw_obj).__name__}"
        )
    objectives = {}
    for side, obj_data in raw_obj.items():
        if not isinstance(obj_data, dict):
            raise ValueError(f"{source_name}: objectives.{side} must be a mapping")
        objectives[side] = ObjectiveOverride(
            label=obj_data.get("label", ""),
            urgency_base=obj_data.get("urgency_base"),
        )

    # Optional: stakes
    stakes = None
    if "stakes" in data and data["stakes"] is not None:
        raw_s = data["stakes"]
        stakes = NarrativeStakes(
            home=int(raw_s.get("home", 0)),
            away=int(raw_s.get("away", 0)),
            notes=raw_s.get("notes"),
        )

    # Optional: rotations
    rotations = None
    if "rotations" in data and data["rotations"] is not None:
        raw_r = data["rotations"]
        rotations = {k: int(v) for k, v in raw_r.items()}

    return Narrative(
        match=match,
        confidence_level=int(data["confidence_level"]),
        objectives=objectives,
        stakes=stakes,
        rotations=rotations,
        intensity_override=_optional_int(data.get("intensity_override")),
        physicality_bias=_optional_int(data.get("physicality_bias")),
        referee_factor=_optional_int(data.get("referee_factor")),
        special_flags=list(data.get("special_flags") or []),
        notes=data.get("notes"),
    )


def _build_match(data: Any, source_name: str = "<unknown>") -> NarrativeMatch:
    if not isinstance(data, dict):
        raise ValueError(
            f"{source_name}: 'match' must be a mapping, got {type(data).__name__}"
        )
    for required in ("home", "away", "date"):
        if required not in data:
            raise ValueError(
                f"{source_name}: missing required match field '{required}'"
            )
    return NarrativeMatch(
        home=str(data["home"]),
        away=str(data["away"]),
        date=str(data["date"]),
        competition=str(data.get("competition", "La Liga")),
        jornada=_optional_int(data.get("jornada")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _check_unknown_fields(
    data: dict[str, Any],
    allowed: set[str],
    context: str = "",
) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"{context}: unknown field(s) {sorted(unknown)!r}. "
            f"Allowed fields: {sorted(allowed)!r}"
        )
