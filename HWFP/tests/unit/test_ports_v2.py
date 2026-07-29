"""Tests for HWFP v2 ports: PerformanceTracker, CalibrationStore, CLVTracker,
LineMonitor, LineupProvider, NotificationSender.

TDD cycle:
  RED  — ImportError until port files are created.
  GREEN — all assertions pass once implementations exist.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Protocol

import pytest

# ── helpers ────────────────────────────────────────────────────────────────────


def _is_protocol(cls: type) -> bool:
    return getattr(cls, "_is_protocol", False)


def _public_methods(cls: type) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def _param_names(cls: type, method: str) -> tuple[str, ...]:
    sig = inspect.signature(getattr(cls, method))
    return tuple(p for p in sig.parameters if p != "self")


# ── port imports (RED: ImportError until files exist) ─────────────────────────

from HWFP.core.ports.calibration_store import CalibrationStore  # noqa: E402
from HWFP.core.ports.clv_tracker import CLVTracker  # noqa: E402
from HWFP.core.ports.line_monitor import LineMonitor  # noqa: E402
from HWFP.core.ports.lineup_provider import LineupProvider  # noqa: E402
from HWFP.core.ports.notification_sender import NotificationSender  # noqa: E402
from HWFP.core.ports.performance_tracker import PerformanceTracker  # noqa: E402

# ── domain imports ────────────────────────────────────────────────────────────

from HWFP.core.domain.lineup import Lineup  # noqa: E402
from HWFP.core.domain.notification import NotificationPriority  # noqa: E402

# ── port registry ─────────────────────────────────────────────────────────────

_NEW_PORTS = [
    PerformanceTracker,
    CalibrationStore,
    CLVTracker,
    LineMonitor,
    LineupProvider,
    NotificationSender,
]


# ── T1: each port is a typing.Protocol ───────────────────────────────────────


@pytest.mark.parametrize(
    "port_cls", _NEW_PORTS, ids=[c.__name__ for c in _NEW_PORTS]
)
def test_each_port_is_a_protocol(port_cls: type) -> None:
    assert _is_protocol(port_cls), f"{port_cls.__name__} must be typing.Protocol"


# ── T2: each port declares the required methods ───────────────────────────────


@pytest.mark.parametrize(
    "port_cls, expected_methods",
    [
        (
            PerformanceTracker,
            {"record_bet", "update_outcome", "get_records", "get_pending_records"},
        ),
        (
            CalibrationStore,
            {
                "get_current_params",
                "save_calibration",
                "get_history",
                "get_bet_records_for_calibration",
            },
        ),
        (CLVTracker, {"record_closing_line", "compute_clv", "get_avg_clv"}),
        (
            LineMonitor,
            {"get_current_odds", "detect_movements", "get_significant_movements"},
        ),
        (LineupProvider, {"get_lineup"}),
        (NotificationSender, {"send", "send_decision", "send_performance_alert"}),
    ],
    ids=[
        "PerformanceTracker",
        "CalibrationStore",
        "CLVTracker",
        "LineMonitor",
        "LineupProvider",
        "NotificationSender",
    ],
)
def test_port_has_declared_methods(
    port_cls: type, expected_methods: set[str]
) -> None:
    actual = _public_methods(port_cls)
    missing = expected_methods - actual
    assert not missing, f"{port_cls.__name__} is missing methods: {missing}"


# ── T3: parameter names ───────────────────────────────────────────────────────


def test_performance_tracker_record_bet_params() -> None:
    assert _param_names(PerformanceTracker, "record_bet") == ("decision",)


def test_performance_tracker_update_outcome_params() -> None:
    assert _param_names(PerformanceTracker, "update_outcome") == ("bet_id", "outcome")


def test_performance_tracker_get_records_params() -> None:
    assert _param_names(PerformanceTracker, "get_records") == ("last_n",)


def test_performance_tracker_get_pending_records_params() -> None:
    assert _param_names(PerformanceTracker, "get_pending_records") == ()


def test_calibration_store_get_current_params_no_params() -> None:
    assert _param_names(CalibrationStore, "get_current_params") == ()


def test_calibration_store_save_calibration_params() -> None:
    assert _param_names(CalibrationStore, "save_calibration") == ("event",)


def test_calibration_store_get_history_no_params() -> None:
    assert _param_names(CalibrationStore, "get_history") == ()


def test_calibration_store_get_bet_records_for_calibration_params() -> None:
    assert _param_names(CalibrationStore, "get_bet_records_for_calibration") == (
        "last_n",
    )


def test_clv_tracker_record_closing_line_params() -> None:
    assert _param_names(CLVTracker, "record_closing_line") == (
        "bet_id",
        "closing_line",
        "closing_odds",
    )


def test_clv_tracker_compute_clv_params() -> None:
    assert _param_names(CLVTracker, "compute_clv") == ("bet_id",)


def test_clv_tracker_get_avg_clv_params() -> None:
    assert _param_names(CLVTracker, "get_avg_clv") == ("last_n",)


def test_line_monitor_get_current_odds_params() -> None:
    assert _param_names(LineMonitor, "get_current_odds") == ("match_id",)


def test_line_monitor_detect_movements_params() -> None:
    assert _param_names(LineMonitor, "detect_movements") == ("match_id", "since")


def test_line_monitor_get_significant_movements_params() -> None:
    assert _param_names(LineMonitor, "get_significant_movements") == ("since",)


def test_lineup_provider_get_lineup_params() -> None:
    assert _param_names(LineupProvider, "get_lineup") == ("match_id",)


def test_notification_sender_send_params() -> None:
    assert _param_names(NotificationSender, "send") == ("message", "priority")


def test_notification_sender_send_decision_params() -> None:
    assert _param_names(NotificationSender, "send_decision") == ("decision",)


def test_notification_sender_send_performance_alert_params() -> None:
    assert _param_names(NotificationSender, "send_performance_alert") == ("snapshot",)


# ── T4: runtime_checkable isinstance checks ───────────────────────────────────


def test_performance_tracker_isinstance() -> None:
    class Fake:
        def record_bet(self, decision): ...
        def update_outcome(self, bet_id, outcome): ...
        def get_records(self, last_n=None): ...
        def get_pending_records(self): ...

    assert isinstance(Fake(), PerformanceTracker)


def test_calibration_store_isinstance() -> None:
    class Fake:
        def get_current_params(self): ...
        def save_calibration(self, event): ...
        def get_history(self): ...
        def get_bet_records_for_calibration(self, last_n): ...

    assert isinstance(Fake(), CalibrationStore)


def test_clv_tracker_isinstance() -> None:
    class Fake:
        def record_closing_line(self, bet_id, closing_line, closing_odds): ...
        def compute_clv(self, bet_id): ...
        def get_avg_clv(self, last_n=None): ...

    assert isinstance(Fake(), CLVTracker)


def test_line_monitor_isinstance() -> None:
    class Fake:
        def get_current_odds(self, match_id): ...
        def detect_movements(self, match_id, since): ...
        def get_significant_movements(self, since): ...

    assert isinstance(Fake(), LineMonitor)


def test_lineup_provider_isinstance() -> None:
    class Fake:
        def get_lineup(self, match_id): ...

    assert isinstance(Fake(), LineupProvider)


def test_notification_sender_isinstance() -> None:
    class Fake:
        def send(self, message, priority=None): ...
        def send_decision(self, decision): ...
        def send_performance_alert(self, snapshot): ...

    assert isinstance(Fake(), NotificationSender)


# ── T5: Lineup domain entity ──────────────────────────────────────────────────


def test_lineup_is_frozen_dataclass() -> None:
    now = datetime(2026, 1, 1)
    lineup = Lineup(
        match_id="m1",
        home_team_id="t1",
        away_team_id="t2",
        home_starters=("p1", "p2"),
        away_starters=("p3", "p4"),
        confirmed_at=now,
    )
    assert lineup.match_id == "m1"
    assert lineup.home_starters == ("p1", "p2")
    with pytest.raises((AttributeError, TypeError)):
        lineup.match_id = "changed"  # type: ignore[misc]


# ── T6: NotificationPriority domain enum ─────────────────────────────────────


def test_notification_priority_values() -> None:
    assert NotificationPriority.LOW == "low"
    assert NotificationPriority.NORMAL == "normal"
    assert NotificationPriority.HIGH == "high"
    assert NotificationPriority.CRITICAL == "critical"


def test_notification_priority_is_str_enum() -> None:
    assert issubclass(NotificationPriority, str)
