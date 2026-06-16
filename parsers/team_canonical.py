"""
parsers.team_canonical — Fuzzy team name canonicalization.

Resolves free-text team names (from user pastes) to their canonical
La Liga 2025-26 form using aliases and difflib fuzzy matching.
No external dependencies — stdlib difflib only.
"""

from __future__ import annotations

import difflib
import unicodedata

__all__ = ["COMMON_ALIASES", "LALIGA_2025_26_TEAMS", "canonicalize"]

# ---------------------------------------------------------------------------
# Canonical team names for La Liga 2025-26
# ---------------------------------------------------------------------------

LALIGA_2025_26_TEAMS: list[str] = [
    "Real Madrid",
    "FC Barcelona",
    "Atlético de Madrid",
    "Athletic Club",
    "Real Sociedad",
    "Real Betis",
    "Villarreal",
    "Sevilla",
    "Girona",
    "Mallorca",
    "Rayo Vallecano",
    "Celta de Vigo",
    "Osasuna",
    "Getafe",
    "Alavés",
    "Espanyol",
    "Valladolid",
    "Levante",
    "Oviedo",
    "Elche",
]

# ---------------------------------------------------------------------------
# Common aliases (exact match, case-insensitive lookup)
# ---------------------------------------------------------------------------

COMMON_ALIASES: dict[str, str] = {
    "atleti": "Atlético de Madrid",
    "atletico": "Atlético de Madrid",
    "atlético": "Atlético de Madrid",
    "atletico de madrid": "Atlético de Madrid",
    "barça": "FC Barcelona",
    "barca": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "athletic": "Athletic Club",
    "athletic bilbao": "Athletic Club",
    "betis": "Real Betis",
    "sociedad": "Real Sociedad",
    "celta": "Celta de Vigo",
    "rayo": "Rayo Vallecano",
    "real": "Real Madrid",
    "madrid": "Real Madrid",
}

# Build a lowercase → canonical mapping for the team list (for exact lookups)
_LOWERCASE_TO_CANONICAL: dict[str, str] = {
    team.lower(): team for team in LALIGA_2025_26_TEAMS
}

# Precompute normalized versions for fuzzy matching
_NORMALIZED_TEAMS: dict[str, str] = {}


def _normalize(text: str) -> str:
    """Normalize text: lowercase, strip accents."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


for _team in LALIGA_2025_26_TEAMS:
    _NORMALIZED_TEAMS[_normalize(_team)] = _team


def canonicalize(name: str, threshold: float = 0.85) -> str | None:
    """Resolve a team name to its canonical La Liga 2025-26 form.

    Resolution order:
    1. Alias lookup (exact, case-insensitive, stripped).
    2. Exact match in LALIGA_2025_26_TEAMS (case-insensitive).
    3. difflib fuzzy match against normalized team names.
    4. Return None if no match above threshold.

    Args:
        name: Raw team name from user input.
        threshold: Minimum similarity score for fuzzy matching. Default: 0.85.

    Returns:
        Canonical team name string, or None if no confident match.
    """
    if not name or not name.strip():
        return None

    stripped = name.strip()
    lower = stripped.lower()

    # 1. Alias lookup
    if lower in COMMON_ALIASES:
        return COMMON_ALIASES[lower]

    # 2. Exact case-insensitive match
    if lower in _LOWERCASE_TO_CANONICAL:
        return _LOWERCASE_TO_CANONICAL[lower]

    # 3. Fuzzy match on normalized names
    normalized_input = _normalize(stripped)
    candidates = list(_NORMALIZED_TEAMS.keys())

    matches = difflib.get_close_matches(
        normalized_input,
        candidates,
        n=1,
        cutoff=threshold,
    )
    if matches:
        return _NORMALIZED_TEAMS[matches[0]]

    return None
