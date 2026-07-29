"""NotificationSender port — dispatch alerts and decision notifications."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from HWFP.core.domain.betting_decision import BettingDecision
from HWFP.core.domain.notification import NotificationPriority
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot


@runtime_checkable
class NotificationSender(Protocol):
    """Send notifications about decisions and performance alerts."""

    def send(
        self,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> None: ...
    def send_decision(self, decision: BettingDecision) -> None: ...
    def send_performance_alert(self, snapshot: PerformanceSnapshot) -> None: ...
