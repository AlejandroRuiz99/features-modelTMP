from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LineMovement:
    match_id: str
    bookmaker: str
    market: str
    line_before: float
    line_after: float
    odds_before: float
    odds_after: float
    delta: float
    detected_at: datetime
    is_significant: bool
