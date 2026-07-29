"""Domain v2 entity tests — TDD RED/GREEN for 6 new entities."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.bet_record import BetOutcome, BetRecord
from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.calibration import (
    CalibrationEvent,
    CalibrationParams,
    CalibrationStatus,
)
from HWFP.core.domain.confidence_score import ConfidenceLevel, ConfidenceScore
from HWFP.core.domain.line_movement import LineMovement
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot

_DT = datetime(2026, 7, 1, 20, 0, 0)


# --- helpers ------------------------------------------------------------


def _make_confidence_score(
    level: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> ConfidenceScore:
    multipliers = {
        ConfidenceLevel.HIGH: 1.0,
        ConfidenceLevel.MEDIUM: 0.6,
        ConfidenceLevel.LOW: 0.25,
    }
    return ConfidenceScore(
        pmf_entropy=0.5,
        referee_sample_size=12,
        feature_fallback_count=0,
        kelly_multiplier=multipliers[level],
        level=level,
    )


def _make_betting_decision() -> BettingDecision:
    return BettingDecision(
        match_id="M1",
        recommendation=Recommendation.BET,
        market="total_fouls_over_21.5",
        line=21.5,
        side="over",
        best_bookmaker="1xbet",
        best_odds_value=1.92,
        p_model=0.58,
        edge=0.1136,
        stake_euros=10.0,
        confidence=_make_confidence_score(),
        reasons=("high edge", "referee profile"),
        generated_at=_DT,
    )


def _make_calibration_params() -> CalibrationParams:
    return CalibrationParams(
        a=1.1,
        b=-0.05,
        n_bets_fitted=200,
        ece_before=0.08,
        ece_after=0.03,
        fitted_at=_DT,
        version=1,
    )


# === ConfidenceScore / ConfidenceLevel ==================================


class TestConfidenceLevel:
    def test_enum_string_values(self) -> None:
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"

    def test_is_str_subclass(self) -> None:
        assert isinstance(ConfidenceLevel.HIGH, str)


class TestConfidenceScore:
    def test_instantiation(self) -> None:
        cs = _make_confidence_score(ConfidenceLevel.MEDIUM)
        assert cs.pmf_entropy == 0.5
        assert cs.referee_sample_size == 12
        assert cs.feature_fallback_count == 0
        assert cs.kelly_multiplier == 0.6
        assert cs.level == ConfidenceLevel.MEDIUM

    def test_frozen(self) -> None:
        cs = _make_confidence_score()
        with pytest.raises(AttributeError):
            cs.pmf_entropy = 9.9  # type: ignore[misc]

    def test_low_confidence_multiplier(self) -> None:
        cs = _make_confidence_score(ConfidenceLevel.LOW)
        assert cs.kelly_multiplier == 0.25


# === BettingDecision / Recommendation ===================================


class TestRecommendation:
    def test_enum_string_values(self) -> None:
        assert Recommendation.BET == "bet"
        assert Recommendation.SKIP == "skip"

    def test_is_str_subclass(self) -> None:
        assert isinstance(Recommendation.BET, str)


class TestBettingDecision:
    def test_instantiation(self) -> None:
        bd = _make_betting_decision()
        assert bd.match_id == "M1"
        assert bd.recommendation == Recommendation.BET
        assert bd.market == "total_fouls_over_21.5"
        assert bd.line == 21.5
        assert bd.side == "over"
        assert bd.best_bookmaker == "1xbet"
        assert bd.best_odds_value == 1.92
        assert bd.p_model == 0.58
        assert bd.stake_euros == 10.0
        assert isinstance(bd.confidence, ConfidenceScore)
        assert bd.reasons == ("high edge", "referee profile")
        assert bd.generated_at == _DT

    def test_frozen(self) -> None:
        bd = _make_betting_decision()
        with pytest.raises(AttributeError):
            bd.match_id = "mutated"  # type: ignore[misc]

    def test_reasons_is_tuple(self) -> None:
        bd = _make_betting_decision()
        assert isinstance(bd.reasons, tuple)

    def test_skip_recommendation(self) -> None:
        bd = BettingDecision(
            match_id="M2",
            recommendation=Recommendation.SKIP,
            market="total_fouls_over_21.5",
            line=21.5,
            side="over",
            best_bookmaker="bet365",
            best_odds_value=1.80,
            p_model=0.45,
            edge=-0.01,
            stake_euros=0.0,
            confidence=_make_confidence_score(ConfidenceLevel.LOW),
            reasons=("low edge",),
            generated_at=_DT,
        )
        assert bd.recommendation == Recommendation.SKIP


# === BetRecord / BetOutcome =============================================


class TestBetOutcome:
    def test_enum_string_values(self) -> None:
        assert BetOutcome.WIN == "win"
        assert BetOutcome.LOSS == "loss"
        assert BetOutcome.VOID == "void"
        assert BetOutcome.PENDING == "pending"

    def test_is_str_subclass(self) -> None:
        assert isinstance(BetOutcome.WIN, str)


class TestBetRecord:
    def test_instantiation_pending(self) -> None:
        br = BetRecord(
            bet_id="bet-001",
            decision=_make_betting_decision(),
            placed_at=_DT,
            outcome=BetOutcome.PENDING,
            closing_line=None,
            closing_odds=None,
            clv=None,
            profit_euros=None,
        )
        assert br.bet_id == "bet-001"
        assert br.outcome == BetOutcome.PENDING
        assert br.closing_line is None
        assert br.clv is None
        assert br.profit_euros is None

    def test_instantiation_settled(self) -> None:
        br = BetRecord(
            bet_id="bet-002",
            decision=_make_betting_decision(),
            placed_at=_DT,
            outcome=BetOutcome.WIN,
            closing_line=21.5,
            closing_odds=1.85,
            clv=0.037,
            profit_euros=9.2,
        )
        assert br.outcome == BetOutcome.WIN
        assert br.profit_euros == 9.2
        assert br.clv == pytest.approx(0.037)

    def test_frozen(self) -> None:
        br = BetRecord(
            bet_id="bet-003",
            decision=_make_betting_decision(),
            placed_at=_DT,
            outcome=BetOutcome.PENDING,
            closing_line=None,
            closing_odds=None,
            clv=None,
            profit_euros=None,
        )
        with pytest.raises(AttributeError):
            br.outcome = BetOutcome.WIN  # type: ignore[misc]


# === LineMovement =======================================================


class TestLineMovement:
    def test_instantiation(self) -> None:
        lm = LineMovement(
            match_id="M1",
            bookmaker="bet365",
            market="total_fouls_over_under",
            line_before=21.5,
            line_after=22.0,
            odds_before=1.95,
            odds_after=1.80,
            delta=0.5,
            detected_at=_DT,
            is_significant=True,
        )
        assert lm.match_id == "M1"
        assert lm.delta == 0.5
        assert lm.is_significant is True

    def test_not_significant_small_delta(self) -> None:
        lm = LineMovement(
            match_id="M1",
            bookmaker="bet365",
            market="total_fouls_over_under",
            line_before=21.5,
            line_after=21.75,
            odds_before=1.95,
            odds_after=1.90,
            delta=0.25,
            detected_at=_DT,
            is_significant=False,
        )
        assert lm.is_significant is False
        assert abs(lm.delta) < 0.5

    def test_frozen(self) -> None:
        lm = LineMovement(
            match_id="M1",
            bookmaker="bet365",
            market="total_fouls_over_under",
            line_before=21.5,
            line_after=22.0,
            odds_before=1.95,
            odds_after=1.80,
            delta=0.5,
            detected_at=_DT,
            is_significant=True,
        )
        with pytest.raises(AttributeError):
            lm.delta = 9.9  # type: ignore[misc]


# === CalibrationParams / CalibrationEvent / CalibrationStatus ===========


class TestCalibrationStatus:
    def test_enum_string_values(self) -> None:
        assert CalibrationStatus.GREEN == "green"
        assert CalibrationStatus.YELLOW == "yellow"
        assert CalibrationStatus.ORANGE == "orange"
        assert CalibrationStatus.RED == "red"

    def test_is_str_subclass(self) -> None:
        assert isinstance(CalibrationStatus.GREEN, str)


class TestCalibrationParams:
    def test_instantiation(self) -> None:
        params = _make_calibration_params()
        assert params.a == 1.1
        assert params.b == -0.05
        assert params.n_bets_fitted == 200
        assert params.ece_before == 0.08
        assert params.ece_after == 0.03
        assert params.version == 1

    def test_frozen(self) -> None:
        params = _make_calibration_params()
        with pytest.raises(AttributeError):
            params.version = 99  # type: ignore[misc]


class TestCalibrationEvent:
    def test_instantiation_accepted(self) -> None:
        event = CalibrationEvent(
            params=_make_calibration_params(),
            trigger="scheduled_30",
            accepted=True,
            recorded_at=_DT,
        )
        assert event.trigger == "scheduled_30"
        assert event.accepted is True

    def test_rejected_when_ece_worse(self) -> None:
        event = CalibrationEvent(
            params=_make_calibration_params(),
            trigger="alarm_ece",
            accepted=False,
            recorded_at=_DT,
        )
        assert event.accepted is False

    def test_frozen(self) -> None:
        event = CalibrationEvent(
            params=_make_calibration_params(),
            trigger="scheduled_30",
            accepted=True,
            recorded_at=_DT,
        )
        with pytest.raises(AttributeError):
            event.accepted = False  # type: ignore[misc]


# === PerformanceSnapshot ================================================


class TestPerformanceSnapshot:
    def _make(self, status: CalibrationStatus) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            status=status,
            n_bets_total=120,
            roi_trailing_30=0.08,
            ece_trailing_50=0.04,
            win_rate_high_conf=0.63,
            clv_avg=0.02,
            kelly_reduction=1.0,
            as_of=_DT,
        )

    def test_instantiation(self) -> None:
        snap = self._make(CalibrationStatus.GREEN)
        assert snap.status == CalibrationStatus.GREEN
        assert snap.n_bets_total == 120
        assert snap.clv_avg == 0.02

    def test_frozen(self) -> None:
        snap = self._make(CalibrationStatus.GREEN)
        with pytest.raises(AttributeError):
            snap.n_bets_total = 999  # type: ignore[misc]

    def test_kelly_multiplier_green(self) -> None:
        snap = self._make(CalibrationStatus.GREEN)
        assert snap.kelly_multiplier_for_status() == 1.0

    def test_kelly_multiplier_yellow(self) -> None:
        snap = self._make(CalibrationStatus.YELLOW)
        assert snap.kelly_multiplier_for_status() == 0.75

    def test_kelly_multiplier_orange(self) -> None:
        snap = self._make(CalibrationStatus.ORANGE)
        assert snap.kelly_multiplier_for_status() == 0.5

    def test_kelly_multiplier_red(self) -> None:
        snap = self._make(CalibrationStatus.RED)
        assert snap.kelly_multiplier_for_status() == 0.0
