"""TelegramNotificationSender — implements NotificationSender via Telegram Bot REST API."""

from __future__ import annotations

import logging

import requests

from HWFP.core.domain.betting_decision import BettingDecision
from HWFP.core.domain.notification import NotificationPriority
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot
from HWFP.serving.telegram.formatters import format_bet_decision, format_performance_snapshot

_LOG = logging.getLogger(__name__)
_TIMEOUT = 10  # seconds


class TelegramNotificationSender:
    """Sends notifications to a Telegram chat via the Bot API.

    Uses requests (sync) so it can be called from both sync use cases and
    async bot handlers without event-loop conflicts.

    Implements the NotificationSender port via duck typing.
    """

    def __init__(self, token: str, chat_id: int) -> None:
        self._chat_id = chat_id
        self._api_url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(
        self,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> None:
        self._post(message)

    def send_decision(self, decision: BettingDecision) -> None:
        self._post(format_bet_decision(decision), parse_mode="Markdown")

    def send_performance_alert(self, snapshot: PerformanceSnapshot) -> None:
        self._post(format_performance_snapshot(snapshot), parse_mode="Markdown")

    def _post(self, text: str, parse_mode: str = "Markdown") -> None:
        try:
            resp = requests.post(
                self._api_url,
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            _LOG.warning("Telegram send failed: %s", exc)
