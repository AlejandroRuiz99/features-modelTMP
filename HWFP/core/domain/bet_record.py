from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from HWFP.core.domain.betting_decision import BettingDecision


class BetOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    VOID = "void"
    PENDING = "pending"


@dataclass(frozen=True)
class BetRecord:
    bet_id: str
    decision: BettingDecision
    placed_at: datetime
    outcome: BetOutcome
    closing_line: float | None
    closing_odds: float | None
    clv: float | None
    profit_euros: float | None
