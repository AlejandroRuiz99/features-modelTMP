"""SchedulerRunner — APScheduler jobs for automated scanning and monitoring."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler

_LOG = logging.getLogger(__name__)


class SchedulerRunner:
    """Wraps APScheduler with the three recurring HWFP jobs.

    All callables are injected so this class has zero domain knowledge
    and is trivially testable by passing spy functions.

    Jobs:
        scan_fn         — runs once daily at `scan_hour` (Madrid time)
        monitor_fn      — runs every `monitor_interval_minutes` minutes
        stats_update_fn — runs once daily before the scan (at scan_hour - 1)
    """

    def __init__(
        self,
        scan_fn: Callable[[], None],
        monitor_fn: Callable[[], None],
        stats_update_fn: Callable[[], None],
        scan_hour: int = 10,
        monitor_interval_minutes: int = 30,
        timezone_str: str = "Europe/Madrid",
    ) -> None:
        self._scan_fn = scan_fn
        self._monitor_fn = monitor_fn
        self._stats_update_fn = stats_update_fn
        self._scan_hour = scan_hour
        self._monitor_interval = monitor_interval_minutes
        self._tz = timezone_str
        self._scheduler: BackgroundScheduler | None = None

    def start(self) -> None:
        self._scheduler = BackgroundScheduler(timezone=self._tz)
        stats_hour = max(0, self._scan_hour - 1)

        self._scheduler.add_job(
            self._stats_update_fn,
            trigger="cron",
            hour=stats_hour,
            minute=0,
            id="stats_update",
        )
        self._scheduler.add_job(
            self._scan_fn,
            trigger="cron",
            hour=self._scan_hour,
            minute=0,
            id="scan_jornada",
        )
        self._scheduler.add_job(
            self._monitor_fn,
            trigger="interval",
            minutes=self._monitor_interval,
            id="monitor_odds",
        )
        self._scheduler.start()
        _LOG.info(
            "Scheduler started: scan@%02d:00, monitor every %dmin (tz=%s)",
            self._scan_hour,
            self._monitor_interval,
            self._tz,
        )

    def shutdown(self, wait: bool = True) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=wait)
            _LOG.info("Scheduler stopped.")
