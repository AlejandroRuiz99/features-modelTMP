"""
parsers.matches_parser — Free-text match list → ParsedMatchday.

Parses user-pasted match lists (Day header + "Home - Away, Referee" lines)
into structured ParsedMatch dataclasses. Uses team_canonical for fuzzy
team name resolution. Pure function — no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from parsers.team_canonical import canonicalize

__all__ = ["ParseError", "ParsedMatch", "ParsedMatchday", "parse_matches_text"]

# ---------------------------------------------------------------------------
# Day/jornada header patterns
# ---------------------------------------------------------------------------

_DAY_NAMES_ES = r"Lunes|Martes|Miércoles|Miercoles|Jueves|Viernes|Sábado|Sabado|Domingo"
_HEADER_RE = re.compile(
    rf"^(?:{_DAY_NAMES_ES}|J\d+|Jornada\s*\d+|\d{{4}}-\d{{2}}-\d{{2}})",
    re.IGNORECASE,
)
_JORNADA_RE = re.compile(r"J(?:ornada\s*)?(\d+)", re.IGNORECASE)

# Match line: "Home - Away, Referee" or "Home \u2013 Away, Referee" or "Home - Away"
_MATCH_LINE_RE = re.compile(
    r"""^\s*
    (?P<home>.+?)\s*           # home team (non-greedy)
    [-\u2013]+\s*              # separator: dash or en-dash
    (?P<away>.+?)              # away team
    (?:\s*,\s*(?P<referee>.+))? # optional referee after comma
    \s*$""",
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParsedMatch:
    """A single parsed match."""

    home: str
    away: str
    referee: str | None
    date: str | None  # YYYY-MM-DD if inferable, else None
    slug: str  # auto-generated
    canonical_match: bool = True  # False if any team name could not be resolved
    kickoff: str | None = None


@dataclass
class ParsedMatchday:
    """Result of parsing a full matchday paste."""

    jornada: int | None
    matches: list[ParsedMatch]
    warnings: list[str] = field(default_factory=list)


class ParseError(ValueError):
    """Raised when input cannot yield any parseable match lines."""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _normalize_slug(text: str) -> str:
    """Convert team name to slug fragment (lowercase, no accents, hyphens)."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", ascii_str)


def _make_slug(home: str, away: str, date: str | None) -> str:
    """Generate a slug for a match."""
    h = _normalize_slug(home)
    a = _normalize_slug(away)
    if date:
        return f"{h}_vs_{a}_{date}"
    return f"{h}_vs_{a}"


def _extract_jornada(line: str) -> int | None:
    """Extract jornada number from a header line."""
    m = _JORNADA_RE.search(line)
    if m:
        return int(m.group(1))
    return None


def _resolve_team(name: str, warnings: list[str]) -> tuple[str, bool]:
    """Resolve team name via canonicalize().

    Returns (resolved_name, canonical_match).
    If not resolvable, returns (original, False) and appends a warning.
    """
    canonical = canonicalize(name.strip())
    if canonical is not None:
        return canonical, True
    warnings.append(
        f"Team name {name.strip()!r} could not be resolved to a canonical name."
    )
    return name.strip(), False


def parse_matches_text(text: str) -> ParsedMatchday:
    """Parse a free-text match list into structured data.

    Supports:
      - Day/jornada header lines ("Sábado J35", "J35", "2026-05-04")
      - Match lines: "Home - Away, Referee" or "Home \u2013 Away"
      - Multiple days in one paste
      - Missing referee (referee=None)
      - Ambiguous teams (canonical_match=False)

    Args:
        text: Raw user input.

    Returns:
        ParsedMatchday with matches and warnings.

    Raises:
        ParseError: If no parseable match lines are found.
    """
    if not text or not text.strip():
        raise ParseError("Input is empty. Please paste match lines.")

    lines = text.splitlines()
    matches: list[ParsedMatch] = []
    warnings: list[str] = []
    jornada: int | None = None
    current_date: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this is a header line
        if _HEADER_RE.match(stripped):
            j = _extract_jornada(stripped)
            if j is not None:
                jornada = j
            # Check for explicit date in header
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stripped)
            if date_match:
                current_date = date_match.group(1)
            continue

        # Try to parse as a match line
        m = _MATCH_LINE_RE.match(stripped)
        if m is None:
            continue

        raw_home = m.group("home").strip()
        raw_away = m.group("away").strip()
        raw_referee = m.group("referee")
        referee = raw_referee.strip() if raw_referee else None

        home, home_ok = _resolve_team(raw_home, warnings)
        away, away_ok = _resolve_team(raw_away, warnings)
        canonical = home_ok and away_ok

        slug = _make_slug(home, away, current_date)

        matches.append(
            ParsedMatch(
                home=home,
                away=away,
                referee=referee,
                date=current_date,
                slug=slug,
                canonical_match=canonical,
            )
        )

    if not matches:
        raise ParseError(
            "No parseable match lines found. "
            "Expected format: 'Home - Away, Referee' (with optional day header)."
        )

    return ParsedMatchday(jornada=jornada, matches=matches, warnings=warnings)
