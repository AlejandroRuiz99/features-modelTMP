"""MatchProvider port — supply upcoming match configurations for scanning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MatchConfig:
    """Minimal config identifying what to scan for a match."""

    match_id: str
    market: str = "total_fouls"
    side: str = "over"


@runtime_checkable
class MatchProvider(Protocol):
    """Return the set of matches (and markets) to scan for a given date."""

    def get_upcoming_configs(self, date: datetime) -> list[MatchConfig]: ...
