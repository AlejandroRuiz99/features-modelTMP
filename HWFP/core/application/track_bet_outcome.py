"""TrackBetOutcomeUseCase — record final outcome and closing line for a placed bet."""

from __future__ import annotations

from dataclasses import dataclass

from HWFP.core.domain.bet_record import BetOutcome, BetRecord
from HWFP.core.domain.notification import NotificationPriority
from HWFP.core.ports.clv_tracker import CLVTracker
from HWFP.core.ports.notification_sender import NotificationSender
from HWFP.core.ports.performance_tracker import PerformanceTracker

_CLV_ALERT_THRESHOLD = -0.05  # alert when avg CLV (last 20) drops below -5%


@dataclass(frozen=True)
class TrackBetOutcomeInput:
    bet_id: str
    outcome: BetOutcome
    closing_line: float | None = None
    closing_odds: float | None = None


@dataclass(frozen=True)
class TrackBetOutcomeOutput:
    record: BetRecord
    clv: float | None
    alert_sent: bool


class TrackBetOutcomeUseCase:
    def __init__(
        self,
        tracker: PerformanceTracker,
        clv_tracker: CLVTracker,
        notifications: NotificationSender,
    ) -> None:
        self._tracker = tracker
        self._clv = clv_tracker
        self._notifications = notifications

    def execute(self, inp: TrackBetOutcomeInput) -> TrackBetOutcomeOutput:
        self._tracker.update_outcome(inp.bet_id, inp.outcome)
        clv: float | None = None
        if inp.closing_line is not None and inp.closing_odds is not None:
            self._clv.record_closing_line(inp.bet_id, inp.closing_line, inp.closing_odds)
            clv = self._clv.compute_clv(inp.bet_id)

        records = self._tracker.get_records()
        record = next((r for r in records if r.bet_id == inp.bet_id), None)
        if record is None:
            raise ValueError(f"BetRecord {inp.bet_id!r} not found after update")

        alert_sent = False
        avg_clv = self._clv.get_avg_clv(last_n=20)
        if avg_clv is not None and avg_clv < _CLV_ALERT_THRESHOLD:
            msg = (
                f"CLV alert: avg CLV over last 20 bets is {avg_clv:.1%} "
                f"(threshold {_CLV_ALERT_THRESHOLD:.0%})"
            )
            self._notifications.send(msg, NotificationPriority.HIGH)
            alert_sent = True

        return TrackBetOutcomeOutput(record=record, clv=clv, alert_sent=alert_sent)
