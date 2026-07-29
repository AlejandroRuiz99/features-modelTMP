from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Lineup:
    match_id: str
    home_team_id: str
    away_team_id: str
    home_starters: tuple[str, ...]
    away_starters: tuple[str, ...]
    confirmed_at: datetime
