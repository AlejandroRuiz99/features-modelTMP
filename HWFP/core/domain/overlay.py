from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Overlay:
    match_id: str
    market: str
    fair_prob: float
    implied_prob: float
    edge: float
