"""MonitorOddsUseCase — detect and alert on significant line movements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from HWFP.core.domain.line_movement import LineMovement
from HWFP.core.domain.notification import NotificationPriority
from HWFP.core.ports.line_monitor import LineMonitor
from HWFP.core.ports.notification_sender import NotificationSender

_HIGH_PRIORITY_DELTA = 1.0  # |delta| >= 1.0 line → HIGH priority alert


@dataclass(frozen=True)
class MonitorOddsInput:
    since: datetime


@dataclass(frozen=True)
class MonitorOddsOutput:
    movements: list[LineMovement]
    alerts_sent: int


class MonitorOddsUseCase:
    def __init__(
        self,
        line_monitor: LineMonitor,
        notifications: NotificationSender,
    ) -> None:
        self._monitor = line_monitor
        self._notifications = notifications

    def execute(self, inp: MonitorOddsInput) -> MonitorOddsOutput:
        movements = self._monitor.get_significant_movements(inp.since)
        alerts_sent = 0
        for movement in movements:
            priority = (
                NotificationPriority.HIGH
                if abs(movement.delta) >= _HIGH_PRIORITY_DELTA
                else NotificationPriority.NORMAL
            )
            self._notifications.send(_format_movement(movement), priority)
            alerts_sent += 1
        return MonitorOddsOutput(movements=movements, alerts_sent=alerts_sent)


def _format_movement(m: LineMovement) -> str:
    direction = "▲" if m.delta > 0 else "▼"
    return (
        f"[LINE MOVEMENT] {m.match_id} | {m.bookmaker} | {m.market}\n"
        f"  Line: {m.line_before} → {m.line_after} "
        f"({direction}{abs(m.delta):.1f})\n"
        f"  Odds: {m.odds_before:.2f} → {m.odds_after:.2f}"
    )
