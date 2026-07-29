from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrainingExample:
    match_id: str
    features: tuple[float, ...]
    actual_fouls: int
    kickoff: datetime
