from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TeamState:
    team_id: str
    as_of: datetime
    avg_fouls_per_match: float
    avg_fouls_conceded: float
    form_window: int
