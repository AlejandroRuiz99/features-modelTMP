"""bot_main.py — Railway entrypoint: starts APScheduler + Telegram bot polling."""

from __future__ import annotations

import logging
import os
import signal
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_LOG = logging.getLogger(__name__)


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        _LOG.error("Missing required env var: %s", key)
        sys.exit(1)
    return value


def main() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    chat_id = int(_require_env("TELEGRAM_CHAT_ID"))
    bankroll = float(os.environ.get("HWFP_BANKROLL", "200.0"))
    scan_hour = int(os.environ.get("HWFP_SCAN_HOUR", "10"))
    monitor_interval = int(os.environ.get("HWFP_MONITOR_INTERVAL_MINUTES", "30"))

    from datetime import datetime, timezone

    from HWFP.core.application.evaluate_performance import EvaluatePerformanceUseCase
    from HWFP.core.application.monitor_odds import MonitorOddsInput, MonitorOddsUseCase
    from HWFP.core.application.recalibrate import RecalibrateUseCase
    from HWFP.core.application.scan_jornada import ScanJornadaUseCase
    from HWFP.core.application.track_bet_outcome import TrackBetOutcomeUseCase
    from HWFP.serving.adapters.aggregated_odds_provider import AggregatedOddsProvider
    from HWFP.serving.adapters.cgm_odds_stub import CGMOddsStub
    from HWFP.serving.adapters.codere_odds_stub import CodereOddsStub
    from HWFP.serving.adapters.filesystem_model_registry import FilesystemModelRegistry
    from HWFP.serving.adapters.kelly_ev_calculator import KellyEVCalculator
    from HWFP.serving.adapters.kelly_staking_calculator import KellyStakingCalculator
    from HWFP.serving.adapters.list_match_provider import ListMatchProvider
    from HWFP.serving.adapters.onexbet_odds_stub import OnexBetOddsStub
    from HWFP.serving.adapters.pytorch_feature_builder import PytorchFeatureBuilder
    from HWFP.serving.adapters.supabase_performance_tracker import SupabasePerformanceTracker
    from HWFP.serving.adapters.supabase_state_adapter import SupabaseStateAdapter
    from HWFP.serving.adapters.yaml_referee_profiler import YamlRefereeProfiler
    from HWFP.serving.telegram.bot import TelegramBotRunner
    from HWFP.serving.telegram.notification_sender import TelegramNotificationSender
    from HWFP.serving.telegram.scheduler import SchedulerRunner

    # ── Adapters ───────────────────────────────────────────────────────────────
    notif = TelegramNotificationSender(token=token, chat_id=chat_id)
    states = SupabaseStateAdapter()
    multi_odds = AggregatedOddsProvider([CodereOddsStub(), CGMOddsStub(), OnexBetOddsStub()])
    model_registry = FilesystemModelRegistry()
    feature_builder = PytorchFeatureBuilder(state_provider_fn=states.get_team_state)
    referee = YamlRefereeProfiler()
    ev_calc = KellyEVCalculator()
    staking = KellyStakingCalculator()
    tracker = SupabasePerformanceTracker()
    match_provider = ListMatchProvider()

    clv_tracker = _InMemoryCLVTracker()     # replace with real impl when available
    calib_store = _InMemoryCalibrationStore()  # replace with SupabaseCalibrationStore

    # ── Use cases ──────────────────────────────────────────────────────────────
    clock = lambda: datetime.now(timezone.utc)

    scan_uc = ScanJornadaUseCase(
        states=states,
        features=feature_builder,
        model_registry=model_registry,
        referee=referee,
        multi_odds=multi_odds,
        ev_calc=ev_calc,
        staking=staking,
        clock=clock,
    )
    evaluate_uc = EvaluatePerformanceUseCase(
        tracker=tracker, clv_tracker=clv_tracker, clock=clock
    )
    recalibrate_uc = RecalibrateUseCase(calibration_store=calib_store, clock=clock)
    track_uc = TrackBetOutcomeUseCase(
        tracker=tracker, clv_tracker=clv_tracker, notifications=notif
    )
    monitor_uc = MonitorOddsUseCase(
        line_monitor=_NullLineMonitor(), notifications=notif
    )

    # ── Bot ────────────────────────────────────────────────────────────────────
    bot = TelegramBotRunner(
        token=token,
        allowed_chat_id=chat_id,
        scan_uc=scan_uc,
        evaluate_uc=evaluate_uc,
        recalibrate_uc=recalibrate_uc,
        track_uc=track_uc,
        performance_tracker=tracker,
        match_provider=match_provider,
        multi_odds=multi_odds,
        bankroll=bankroll,
        clock=clock,
    )

    # ── Scheduler ──────────────────────────────────────────────────────────────
    def _scheduled_scan() -> None:
        _LOG.info("Scheduled scan triggered")
        bot._run_scan([])  # noqa: SLF001

    def _scheduled_monitor() -> None:
        monitor_uc.execute(MonitorOddsInput(since=datetime.now(timezone.utc)))

    def _scheduled_stats_update() -> None:
        _LOG.info("Stats update placeholder — implement SupabaseStatsUpdater")

    scheduler = SchedulerRunner(
        scan_fn=_scheduled_scan,
        monitor_fn=_scheduled_monitor,
        stats_update_fn=_scheduled_stats_update,
        scan_hour=scan_hour,
        monitor_interval_minutes=monitor_interval,
    )
    scheduler.start()

    def _shutdown(sig, frame):  # noqa: ANN001
        _LOG.info("Shutting down (signal %s)", sig)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _LOG.info("Starting Telegram bot (bankroll=€%.0f, chat_id=%d)", bankroll, chat_id)
    bot.run_polling()


class _InMemoryCLVTracker:
    def record_closing_line(self, bet_id: str, closing_line: float, closing_odds: float) -> None:
        pass

    def compute_clv(self, bet_id: str) -> float | None:
        return None

    def get_avg_clv(self, last_n: int | None = None) -> float | None:
        return None


class _InMemoryCalibrationStore:
    def get_current_params(self):
        return None

    def save_calibration(self, event) -> None:
        pass

    def get_history(self) -> list:
        return []

    def get_bet_records_for_calibration(self, last_n: int) -> list:
        return []


class _NullLineMonitor:
    """Placeholder until a real LineMonitor adapter is implemented."""

    def get_current_odds(self, match_id: str):
        return []

    def detect_movements(self, match_id: str, since):
        return []

    def get_significant_movements(self, since):
        return []


if __name__ == "__main__":
    main()
