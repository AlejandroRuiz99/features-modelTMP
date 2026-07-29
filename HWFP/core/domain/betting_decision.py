from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from HWFP.core.domain.confidence_score import ConfidenceScore


class Recommendation(str, Enum):
    BET = "bet"
    SKIP = "skip"


@dataclass(frozen=True)
class BettingDecision:
    match_id: str
    recommendation: Recommendation
    market: str
    line: float
    side: str
    best_bookmaker: str
    best_odds_value: float
    p_model: float
    edge: float
    stake_euros: float
    confidence: ConfidenceScore
    reasons: tuple[str, ...]
    generated_at: datetime
