"""bot_main.py — Railway entrypoint: starts APScheduler + Telegram bot polling."""

from __future__ import annotations

import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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


def _resolve_checkpoints_dir() -> Path:
    """Resolve the production checkpoints dir: env override, else package default.

    Pure function (no side effects beyond reading the environment) so it is
    testable in isolation from `main()`, which also reads Telegram env vars
    and blocks on `bot.run_polling()`.
    """
    from HWFP.models.paths import default_checkpoints_dir

    override = os.environ.get("HWFP_CHECKPOINTS_DIR")
    return Path(override) if override else default_checkpoints_dir()


@dataclass
class BotContainer:
    """Fully-wired runtime container returned by `build_container()`.

    `bot`/`scheduler` are what `main()` needs to start the process;
    `feature_builder`/`model_registry` are exposed so tests can exercise a
    feature-build + predict round trip without reaching into `bot`'s
    private use-case attributes.
    """

    bot: Any
    scheduler: Any
    feature_builder: Any
    model_registry: Any


def _unavailable_partidos_source() -> list[dict]:
    """Placeholder data source for HWFP.features.core.state_cache.

    No production Supabase-backed "raw partidos" fetcher exists yet — the
    absorbed feature pipeline (design D1) expects one to be injected via
    `state_cache.set_data_source()` at composition time, but building a real
    one requires re-deriving the exact row shape the legacy
    `features_generator/selection` package used to fetch (dropped in PR2,
    per design). Tracked as a known gap; replace with a real implementation
    when available (see HWFP/features/core/state_cache.py's set_data_source
    contract).
    """
    raise RuntimeError(
        "No production partidos data source wired yet — replace "
        "_unavailable_partidos_source with a real Supabase-backed fetcher."
    )


def build_container(
    *,
    token: str,
    chat_id: int,
    bankroll: float = 200.0,
    scan_hour: int = 10,
    monitor_interval_minutes: int = 30,
    checkpoints_dir: Path | None = None,
    partidos_source: Callable[[], list[dict]] | None = None,
    states: Any = None,
    model_registry: Any = None,
    feature_builder: Any = None,
    referee: Any = None,
    ev_calc: Any = None,
    staking: Any = None,
    tracker: Any = None,
    match_provider: Any = None,
    multi_odds: Any = None,
    clv_tracker: Any = None,
    calib_store: Any = None,
    line_monitor: Any = None,
    clock: Callable[[], datetime] | None = None,
    bot: Any = None,
    scheduler: Any = None,
) -> BotContainer:
    """Wire every adapter, use case, and the Telegram bot into a runnable container.

    Pure factory: makes no Telegram network calls and reads no environment
    variables itself — every credential or config value the caller (`main()`)
    would otherwise derive from `os.environ` must already be resolved and
    passed in. Every adapter parameter defaults to the real production
    adapter when omitted, so `main()` only needs to supply `token`/`chat_id`/
    `partidos_source`; tests override individual ports with fakes.

    Also wires `HWFP.features.core.state_cache.set_data_source(partidos_source)`
    at composition time — required alongside `state_provider_fn=state_cache.get_state`,
    since the absorbed feature pipeline's state cache has no data source of
    its own (design D1).

    `bot`/`scheduler` are injectable too: when omitted, the real
    `TelegramBotRunner`/`SchedulerRunner` are constructed (requires the
    optional `python-telegram-bot`/`apscheduler` packages); tests that only
    care about feature-build/predict wiring can inject lightweight
    placeholders to avoid that dependency entirely.
    """
    from HWFP.core.application.evaluate_performance import EvaluatePerformanceUseCase
    from HWFP.core.application.monitor_odds import MonitorOddsInput, MonitorOddsUseCase
    from HWFP.core.application.recalibrate import RecalibrateUseCase
    from HWFP.core.application.scan_jornada import ScanJornadaUseCase
    from HWFP.core.application.track_bet_outcome import TrackBetOutcomeUseCase
    from HWFP.features.core import state_cache
    from HWFP.serving.adapters.aggregated_odds_provider import AggregatedOddsProvider
    from HWFP.serving.adapters.cgm_odds_stub import CGMOddsStub
    from HWFP.serving.adapters.codere_odds_stub import CodereOddsStub
    from HWFP.serving.adapters.filesystem_model_registry import FilesystemModelRegistry
    from HWFP.serving.adapters.kelly_ev_calculator import KellyEVCalculator
    from HWFP.serving.adapters.kelly_staking_calculator import KellyStakingCalculator
    from HWFP.serving.adapters.list_match_provider import ListMatchProvider
    from HWFP.serving.adapters.onexbet_odds_stub import OnexBetOddsStub
    from HWFP.serving.adapters.pytorch_feature_builder import PyTorchFeatureBuilder
    from HWFP.serving.adapters.supabase_performance_tracker import SupabasePerformanceTracker
    from HWFP.serving.adapters.supabase_state_adapter import SupabaseStateAdapter
    from HWFP.serving.adapters.yaml_referee_profiler import YamlRefereeProfiler
    from HWFP.serving.telegram.notification_sender import TelegramNotificationSender

    state_cache.set_data_source(partidos_source or _unavailable_partidos_source)

    resolved_checkpoints_dir = checkpoints_dir if checkpoints_dir is not None else _resolve_checkpoints_dir()
    clock = clock or (lambda: datetime.now(timezone.utc))

    # ── Adapters ───────────────────────────────────────────────────────────────
    notif = TelegramNotificationSender(token=token, chat_id=chat_id)
    states = states if states is not None else SupabaseStateAdapter()
    multi_odds = multi_odds if multi_odds is not None else AggregatedOddsProvider(
        [CodereOddsStub(), CGMOddsStub(), OnexBetOddsStub()]
    )
    model_registry = model_registry if model_registry is not None else FilesystemModelRegistry(
        checkpoints_dir=resolved_checkpoints_dir
    )
    feature_builder = feature_builder if feature_builder is not None else PyTorchFeatureBuilder(
        state_provider_fn=state_cache.get_state
    )
    referee = referee if referee is not None else YamlRefereeProfiler()
    ev_calc = ev_calc if ev_calc is not None else KellyEVCalculator()
    staking = staking if staking is not None else KellyStakingCalculator()
    tracker = tracker if tracker is not None else SupabasePerformanceTracker()
    match_provider = match_provider if match_provider is not None else ListMatchProvider()

    clv_tracker = clv_tracker if clv_tracker is not None else _InMemoryCLVTracker()
    calib_store = calib_store if calib_store is not None else _InMemoryCalibrationStore()
    line_monitor = line_monitor if line_monitor is not None else _NullLineMonitor()

    # ── Use cases ──────────────────────────────────────────────────────────────
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
        line_monitor=line_monitor, notifications=notif
    )

    # ── Bot ────────────────────────────────────────────────────────────────────
    if bot is None:
        from HWFP.serving.telegram.bot import TelegramBotRunner

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
    if scheduler is None:
        from HWFP.serving.telegram.scheduler import SchedulerRunner

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
            monitor_interval_minutes=monitor_interval_minutes,
        )

    return BotContainer(
        bot=bot,
        scheduler=scheduler,
        feature_builder=feature_builder,
        model_registry=model_registry,
    )


def main() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    chat_id = int(_require_env("TELEGRAM_CHAT_ID"))
    bankroll = float(os.environ.get("HWFP_BANKROLL", "200.0"))
    scan_hour = int(os.environ.get("HWFP_SCAN_HOUR", "10"))
    monitor_interval = int(os.environ.get("HWFP_MONITOR_INTERVAL_MINUTES", "30"))

    container = build_container(
        token=token,
        chat_id=chat_id,
        bankroll=bankroll,
        scan_hour=scan_hour,
        monitor_interval_minutes=monitor_interval,
        checkpoints_dir=_resolve_checkpoints_dir(),
    )
    bot = container.bot
    scheduler = container.scheduler

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
