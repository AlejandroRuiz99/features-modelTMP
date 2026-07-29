"""PerformanceTracker port — record and query bet records."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from HWFP.core.domain.bet_record import BetOutcome, BetRecord
from HWFP.core.domain.betting_decision import BettingDecision


@runtime_checkable
class PerformanceTracker(Protocol):
    """Persist and retrieve bet records for performance analysis."""

    def record_bet(self, decision: BettingDecision) -> BetRecord: ...
    def update_outcome(self, bet_id: str, outcome: BetOutcome) -> None: ...
    def get_records(self, last_n: int | None = None) -> list[BetRecord]: ...
    def get_pending_records(self) -> list[BetRecord]: ...
