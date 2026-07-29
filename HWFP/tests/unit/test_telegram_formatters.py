"""Tests for pure Telegram formatting functions."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.calibration import CalibrationStatus
from HWFP.core.domain.confidence_score import ConfidenceLevel, ConfidenceScore
from HWFP.core.domain.line_movement import LineMovement
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot
from HWFP.serving.telegram.formatters import (
    format_bet_decision,
    format_line_movement,
    format_performance_snapshot,
    format_scan_summary,
)
from HWFP.serving.telegram.bot import _parse_settle_args
from HWFP.core.domain.bet_record import BetOutcome

_NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def _make_confidence(level: ConfidenceLevel = ConfidenceLevel.HIGH) -> ConfidenceScore:
    mult = {ConfidenceLevel.HIGH: 1.0, ConfidenceLevel.MEDIUM: 0.6, ConfidenceLevel.LOW: 0.25}[level]
    return ConfidenceScore(
        pmf_entropy=1.5,
        referee_sample_size=25,
        feature_fallback_count=0,
        kelly_multiplier=mult,
        level=level,
    )


def _make_decision(
    match_id: str = "laliga-2025-38-001",
    recommendation: Recommendation = Recommendation.BET,
    edge: float = 0.13,
) -> BettingDecision:
    return BettingDecision(
        match_id=match_id,
        recommendation=recommendation,
        market="total_fouls",
        line=9.5,
        side="over",
        best_bookmaker="codere",
        best_odds_value=1.95,
        p_model=0.58,
        edge=edge,
        stake_euros=25.0,
        confidence=_make_confidence(),
        reasons=(f"edge={edge:.1%}", "p_model=58.0%"),
        generated_at=_NOW,
    )


def _make_snapshot(status: CalibrationStatus = CalibrationStatus.GREEN) -> PerformanceSnapshot:
    return PerformanceSnapshot(
        status=status,
        n_bets_total=72,
        roi_trailing_30=0.06,
        ece_trailing_50=0.03,
        win_rate_high_conf=0.61,
        clv_avg=0.02,
        kelly_reduction=0.0,
        as_of=_NOW,
    )


def _make_movement(delta: float = 0.5) -> LineMovement:
    return LineMovement(
        match_id="laliga-2025-38-001",
        bookmaker="codere",
        market="total_fouls",
        line_before=9.5,
        line_after=9.5 + delta,
        odds_before=1.90,
        odds_after=1.80,
        delta=delta,
        detected_at=_NOW,
        is_significant=True,
    )


# ── format_bet_decision ────────────────────────────────────────────────────────


class TestFormatBetDecision:
    def test_contains_match_id(self):
        msg = format_bet_decision(_make_decision())
        assert "laliga-2025-38-001" in msg

    def test_contains_edge(self):
        msg = format_bet_decision(_make_decision(edge=0.13))
        assert "13" in msg  # 13.0% in some form

    def test_contains_stake(self):
        msg = format_bet_decision(_make_decision())
        assert "25" in msg

    def test_contains_bookmaker(self):
        msg = format_bet_decision(_make_decision())
        assert "codere" in msg

    def test_contains_odds(self):
        msg = format_bet_decision(_make_decision())
        assert "1.95" in msg

    def test_non_empty_string(self):
        msg = format_bet_decision(_make_decision())
        assert len(msg) > 20


# ── format_scan_summary ────────────────────────────────────────────────────────


class TestFormatScanSummary:
    def test_no_bets_returns_no_value_message(self):
        skip = _make_decision(recommendation=Recommendation.SKIP)
        msg = format_scan_summary([skip])
        assert "no value" in msg.lower() or "skip" in msg.lower()

    def test_one_bet_in_summary(self):
        bet = _make_decision()
        msg = format_scan_summary([bet])
        assert "laliga-2025-38-001" in msg

    def test_shows_bet_count(self):
        bets = [_make_decision(f"m{i}") for i in range(3)]
        msg = format_scan_summary(bets)
        assert "3" in msg

    def test_mixed_shows_only_bets(self):
        bet = _make_decision("m1")
        skip = _make_decision("m2", recommendation=Recommendation.SKIP)
        msg = format_scan_summary([bet, skip])
        assert "m1" in msg
        # skip match might not appear prominently
        assert "1" in msg  # 1 BET


# ── format_performance_snapshot ───────────────────────────────────────────────


class TestFormatPerformanceSnapshot:
    def test_contains_status(self):
        msg = format_performance_snapshot(_make_snapshot(CalibrationStatus.GREEN))
        assert "GREEN" in msg.upper()

    def test_contains_n_bets(self):
        msg = format_performance_snapshot(_make_snapshot())
        assert "72" in msg

    def test_contains_roi(self):
        msg = format_performance_snapshot(_make_snapshot())
        assert "6" in msg  # 6% ROI

    def test_contains_ece(self):
        msg = format_performance_snapshot(_make_snapshot())
        assert "0.03" in msg

    def test_red_status_shows_emoji(self):
        msg = format_performance_snapshot(_make_snapshot(CalibrationStatus.RED))
        assert "🔴" in msg

    def test_green_status_shows_emoji(self):
        msg = format_performance_snapshot(_make_snapshot(CalibrationStatus.GREEN))
        assert "🟢" in msg


# ── format_line_movement ───────────────────────────────────────────────────────


class TestFormatLineMovement:
    def test_contains_match_id(self):
        msg = format_line_movement(_make_movement())
        assert "laliga-2025-38-001" in msg

    def test_positive_delta_shows_up_arrow(self):
        msg = format_line_movement(_make_movement(delta=0.5))
        assert "▲" in msg

    def test_negative_delta_shows_down_arrow(self):
        msg = format_line_movement(_make_movement(delta=-0.5))
        assert "▼" in msg

    def test_contains_bookmaker(self):
        msg = format_line_movement(_make_movement())
        assert "codere" in msg


# ── _parse_settle_args ─────────────────────────────────────────────────────────


class TestParseSettleArgs:
    def test_valid_win_no_closing(self):
        result = _parse_settle_args(["bet-123", "win"])
        assert result is not None
        bet_id, outcome, line, odds = result
        assert bet_id == "bet-123"
        assert outcome == BetOutcome.WIN
        assert line is None
        assert odds is None

    def test_valid_loss_with_closing(self):
        result = _parse_settle_args(["bet-456", "loss", "9.5", "1.85"])
        assert result is not None
        bet_id, outcome, line, odds = result
        assert outcome == BetOutcome.LOSS
        assert line == pytest.approx(9.5)
        assert odds == pytest.approx(1.85)

    def test_valid_void(self):
        result = _parse_settle_args(["bet-789", "void"])
        assert result is not None
        assert result[1] == BetOutcome.VOID

    def test_invalid_outcome_returns_none(self):
        assert _parse_settle_args(["bet-1", "draw"]) is None

    def test_too_few_args_returns_none(self):
        assert _parse_settle_args(["bet-1"]) is None

    def test_three_args_returns_none(self):
        # 3 args is ambiguous (bet_id, outcome, one float) — invalid
        assert _parse_settle_args(["bet-1", "win", "9.5"]) is None

    def test_empty_returns_none(self):
        assert _parse_settle_args([]) is None

    def test_non_numeric_closing_returns_none(self):
        assert _parse_settle_args(["bet-1", "win", "abc", "1.85"]) is None

    def test_case_insensitive_outcome(self):
        result = _parse_settle_args(["bet-1", "WIN"])
        assert result is not None
        assert result[1] == BetOutcome.WIN
