"""
overlay.schema — Dataclasses and validators for Narrative YAML.

Defines the data model for match narratives consumed by the overlay system.
All validation is performed in __post_init__ methods — fail fast on bad input.
No external dependencies (stdlib + dataclasses only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "VALID_FLAGS",
    "VALID_LABELS",
    "Narrative",
    "NarrativeMatch",
    "NarrativeStakes",
    "ObjectiveOverride",
]

# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------

VALID_LABELS: frozenset[str] = frozenset(
    {
        "descenso",
        "salvacion",
        "mid",
        "europa",
        "ucl",
        "titulo",
    }
)

VALID_FLAGS: frozenset[str] = frozenset(
    {
        "stakes_both_relegation",
        "stakes_one_relegation",
        "derbi",
        "copa_recent_extra_time",
        "european_midweek",
        "coach_debut",
        "coach_pressure",
        "last_round_drama",
        "weather_extreme",
        "b_team_expected",
        "key_injuries_home",
        "key_injuries_away",
        "morbo",
        "dead_rubber",
        "late_season",
        "early_season",
        "physical_clash",
        "strict_ref_announced",
        "permissive_ref_announced",
    }
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NarrativeMatch:
    """Identity fields for the match being annotated."""

    home: str
    away: str
    date: str  # ISO format YYYY-MM-DD
    competition: str = "La Liga"
    jornada: int | None = None


@dataclass
class ObjectiveOverride:
    """Override for a team's competitive objective and urgency.

    Args:
        label: One of VALID_LABELS.
        urgency_base: Explicit urgency float in [0.0, 1.0]. If None, derived
            from the label via LABEL_URGENCY_MAP in overlay.objective.
    """

    label: str
    urgency_base: float | None = None

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ValueError(
                f"Invalid label {self.label!r}. Valid values: {sorted(VALID_LABELS)}"
            )
        if self.urgency_base is not None and not (0.0 <= self.urgency_base <= 1.0):
            raise ValueError(
                f"urgency_base must be in [0.0, 1.0], got {self.urgency_base}"
            )


@dataclass
class NarrativeStakes:
    """Perceived stakes for each team (0 = none, 5 = maximum)."""

    home: int
    away: int
    notes: str | None = None

    def __post_init__(self) -> None:
        for attr in ("home", "away"):
            val = getattr(self, attr)
            if not (0 <= val <= 5):
                raise ValueError(f"stakes.{attr} must be in [0, 5], got {val}")


@dataclass
class Narrative:
    """Full match narrative consumed by the overlay system.

    Required fields:
        match: NarrativeMatch (home, away, date)
        confidence_level: int in [1, 5]

    All other fields are optional and default to None / [].

    Unknown keyword arguments raise TypeError (dataclass default).
    """

    match: NarrativeMatch
    confidence_level: int

    # Required fields (D17: objectives is now mandatory)
    objectives: dict[str, ObjectiveOverride]
    stakes: NarrativeStakes | None = None
    rotations: dict[str, int] | None = None
    intensity_override: int | None = None
    physicality_bias: int | None = None
    referee_factor: int | None = None
    special_flags: list[str] = field(default_factory=list)
    notes: str | None = None

    def __post_init__(self) -> None:
        self._validate_confidence_level()
        self._validate_physicality_bias()
        self._validate_referee_factor()
        self._validate_intensity_override()
        self._validate_rotations()
        self._validate_special_flags()
        self._validate_objectives_keys()

    def _validate_confidence_level(self) -> None:
        if not (1 <= self.confidence_level <= 5):
            raise ValueError(
                f"confidence_level must be in [1, 5], got {self.confidence_level}"
            )

    def _validate_physicality_bias(self) -> None:
        if self.physicality_bias is not None and not (-2 <= self.physicality_bias <= 2):
            raise ValueError(
                f"physicality_bias must be in [-2, +2], got {self.physicality_bias}"
            )

    def _validate_referee_factor(self) -> None:
        if self.referee_factor is not None and not (-2 <= self.referee_factor <= 2):
            raise ValueError(
                f"referee_factor must be in [-2, +2], got {self.referee_factor}"
            )

    def _validate_intensity_override(self) -> None:
        if self.intensity_override is not None and not (
            0 <= self.intensity_override <= 5
        ):
            raise ValueError(
                f"intensity_override must be in [0, 5], got {self.intensity_override}"
            )

    def _validate_rotations(self) -> None:
        if self.rotations is not None:
            for side in ("home", "away"):
                if side in self.rotations:
                    val = self.rotations[side]
                    if not (0 <= val <= 5):
                        raise ValueError(
                            f"rotations.{side} must be in [0, 5], got {val}"
                        )

    def _validate_special_flags(self) -> None:
        unknown = [f for f in self.special_flags if f not in VALID_FLAGS]
        if unknown:
            raise ValueError(
                f"Unknown special_flags: {unknown!r}. "
                f"Valid flags: {sorted(VALID_FLAGS)}"
            )

    def _validate_objectives_keys(self) -> None:
        unknown_keys = set(self.objectives) - {"home", "away"}
        if unknown_keys:
            raise ValueError(
                f"objectives keys must be 'home'/'away', "
                f"got unknown keys: {unknown_keys}"
            )
        if "home" not in self.objectives:
            raise ValueError("objectives must contain 'home' key")
        if "away" not in self.objectives:
            raise ValueError("objectives must contain 'away' key")
