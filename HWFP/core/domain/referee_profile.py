from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefereeProfile:
    referee_id: str
    avg_fouls_per_match: float
    std_fouls_per_match: float
    sample_size: int
