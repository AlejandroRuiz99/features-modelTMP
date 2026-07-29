"""TelegramBotRunner — bidirectional Telegram bot with all command handlers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from HWFP.core.application.evaluate_performance import EvaluatePerformanceUseCase
from HWFP.core.application.recalibrate import RecalibrateInput, RecalibrateUseCase
from HWFP.core.application.scan_jornada import (
    MatchCandidate,
    ScanJornadaInput,
    ScanJornadaOutput,
    ScanJornadaUseCase,
)
from HWFP.core.application.track_bet_outcome import (
    TrackBetOutcomeInput,
    TrackBetOutcomeUseCase,
)
from HWFP.core.domain.bet_record import BetOutcome
from HWFP.core.ports.match_provider import MatchProvider
from HWFP.core.ports.multi_odds_provider import MultiOddsProvider
from HWFP.core.ports.performance_tracker import PerformanceTracker
from HWFP.serving.telegram.formatters import format_performance_snapshot, format_scan_summary

_LOG = logging.getLogger(__name__)

_VALID_OUTCOMES = {o.value: o for o in BetOutcome}


class TelegramBotRunner:
    """Bidirectional Telegram bot.

    Guards every handler so only messages from `allowed_chat_id` are processed.
    Sync use cases run inside asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(
        self,
        token: str,
        allowed_chat_id: int,
        scan_uc: ScanJornadaUseCase,
        evaluate_uc: EvaluatePerformanceUseCase,
        recalibrate_uc: RecalibrateUseCase,
        track_uc: TrackBetOutcomeUseCase,
        performance_tracker: PerformanceTracker,
        match_provider: MatchProvider,
        multi_odds: MultiOddsProvider,
        bankroll: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._token = token
        self._allowed_chat_id = allowed_chat_id
        self._scan_uc = scan_uc
        self._evaluate_uc = evaluate_uc
        self._recalibrate_uc = recalibrate_uc
        self._track_uc = track_uc
        self._tracker = performance_tracker
        self._match_provider = match_provider
        self._multi_odds = multi_odds
        self._bankroll = bankroll
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # ── Guard ──────────────────────────────────────────────────────────────────

    def _authorized(self, update: Update) -> bool:
        return (
            update.effective_chat is not None
            and update.effective_chat.id == self._allowed_chat_id
        )

    # ── Command handlers ───────────────────────────────────────────────────────

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        text = (
            "*HWFP Betting Bot*\n\n"
            "/scan — Scan today's matches for EV+\n"
            "/scan m1 m2 ... — Scan specific match IDs\n"
            "/status — Performance snapshot\n"
            "/calibrate — Run Platt scaling calibration\n"
            "/settle <bet\\_id> <win|loss|void> — Record outcome\n"
            "/settle <bet\\_id> <win|loss|void> <line> <odds> — With closing line\n"
            "/pending — List pending bets\n"
            "/help — This message"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        await update.message.reply_text("Scanning...")
        args = context.args or []
        result: ScanJornadaOutput = await asyncio.to_thread(self._run_scan, args)
        msg = format_scan_summary(result.decisions)
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        out = await asyncio.to_thread(self._evaluate_uc.execute)
        await update.message.reply_text(
            format_performance_snapshot(out.snapshot), parse_mode="Markdown"
        )

    async def cmd_calibrate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        await update.message.reply_text("Running calibration...")
        out = await asyncio.to_thread(
            lambda: self._recalibrate_uc.execute(RecalibrateInput(trigger="manual"))
        )
        await update.message.reply_text(out.message)

    async def cmd_settle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        args = context.args or []
        parsed = _parse_settle_args(args)
        if parsed is None:
            await update.message.reply_text(
                "Usage: /settle <bet\\_id> <win|loss|void> [<closing\\_line> <closing\\_odds>]",
                parse_mode="Markdown",
            )
            return
        bet_id, outcome, closing_line, closing_odds = parsed
        try:
            await asyncio.to_thread(
                lambda: self._track_uc.execute(
                    TrackBetOutcomeInput(
                        bet_id=bet_id,
                        outcome=outcome,
                        closing_line=closing_line,
                        closing_odds=closing_odds,
                    )
                )
            )
        except (KeyError, ValueError) as exc:
            await update.message.reply_text(f"Error: {exc}")
            return
        emoji = {"win": "✅", "loss": "❌", "void": "↩️"}.get(outcome.value, "")
        clv_note = f" (CLV recorded)" if closing_line is not None else ""
        await update.message.reply_text(
            f"{emoji} `{bet_id}` settled as *{outcome.value}*{clv_note}",
            parse_mode="Markdown",
        )

    async def cmd_pending(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        pending = await asyncio.to_thread(self._tracker.get_pending_records)
        if not pending:
            await update.message.reply_text("No pending bets.")
            return
        lines = [
            f"• `{r.bet_id}` — {r.decision.match_id} "
            f"{r.decision.side} {r.decision.line} "
            f"@ {r.decision.best_odds_value:.2f} ({r.decision.best_bookmaker})"
            f" €{r.decision.stake_euros:.0f}"
            for r in pending
        ]
        await update.message.reply_text(
            f"*Pending bets ({len(pending)}):*\n" + "\n".join(lines),
            parse_mode="Markdown",
        )

    # ── Sync scan helper (runs in thread) ─────────────────────────────────────

    def _run_scan(self, match_id_args: list[str]) -> ScanJornadaOutput:
        if match_id_args:
            candidates = [
                MatchCandidate(match_id=mid, market="total_fouls", line=0.0, side="over")
                for mid in match_id_args
            ]
            # Resolve line from best odds
            resolved: list[MatchCandidate] = []
            for c in candidates:
                best = self._multi_odds.best_odds(c.match_id, c.market)
                if best is not None:
                    resolved.append(
                        MatchCandidate(
                            match_id=c.match_id,
                            market=c.market,
                            line=best.line,
                            side=best.side,
                        )
                    )
            candidates = resolved
        else:
            now = self._clock()
            configs = self._match_provider.get_upcoming_configs(now)
            candidates = []
            for cfg in configs:
                best = self._multi_odds.best_odds(cfg.match_id, cfg.market)
                if best is not None:
                    candidates.append(
                        MatchCandidate(
                            match_id=cfg.match_id,
                            market=cfg.market,
                            line=best.line,
                            side=cfg.side,
                        )
                    )

        if not candidates:
            from HWFP.core.application.scan_jornada import ScanJornadaOutput
            return ScanJornadaOutput(decisions=[])

        snapshot = self._evaluate_uc.execute().snapshot
        return self._scan_uc.execute(
            ScanJornadaInput(candidates=candidates, bankroll=self._bankroll),
            snapshot=snapshot,
        )

    # ── Bot lifecycle ──────────────────────────────────────────────────────────

    def run_polling(self) -> None:
        """Build and start the Application in long-polling mode (blocking)."""
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self.cmd_help))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("scan", self.cmd_scan))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("calibrate", self.cmd_calibrate))
        app.add_handler(CommandHandler("settle", self.cmd_settle))
        app.add_handler(CommandHandler("pending", self.cmd_pending))
        _LOG.info("Bot polling started (allowed_chat_id=%d)", self._allowed_chat_id)
        app.run_polling(allowed_updates=Update.ALL_TYPES)


# ── Pure helpers ───────────────────────────────────────────────────────────────


def _parse_settle_args(
    args: list[str],
) -> tuple[str, BetOutcome, float | None, float | None] | None:
    """Parse /settle args. Returns None on invalid input."""
    if len(args) < 2 or len(args) == 3 or len(args) > 4:
        return None
    bet_id = args[0]
    outcome_str = args[1].lower()
    if outcome_str not in _VALID_OUTCOMES:
        return None
    outcome = _VALID_OUTCOMES[outcome_str]
    if len(args) == 4:
        try:
            closing_line = float(args[2])
            closing_odds = float(args[3])
        except ValueError:
            return None
        return bet_id, outcome, closing_line, closing_odds
    return bet_id, outcome, None, None
