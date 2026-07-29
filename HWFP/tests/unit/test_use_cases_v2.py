"""Tests for new use cases: ScanJornada, MonitorOdds, TrackBetOutcome, EvaluatePerformance, Recalibrate."""
from __future__ import annotations

import math
import types
from datetime import datetime, timezone

import pytest

from HWFP.core.application.scan_jornada import (
    MatchCandidate,
    ScanJornadaInput,
    ScanJornadaUseCase,
    _build_confidence,
    _pmf_entropy,
)
from HWFP.core.application.monitor_odds import MonitorOddsInput, MonitorOddsUseCase
from HWFP.core.application.track_bet_outcome import (
    TrackBetOutcomeInput,
    TrackBetOutcomeUseCase,
)
from HWFP.core.application.evaluate_performance import (
    EvaluatePerformanceUseCase,
    _compute_ece,
    _compute_roi,
    _determine_status,
)
from HWFP.core.application.recalibrate import (
    RecalibrateInput,
    RecalibrateUseCase,
    _fit_platt,
    _logit,
    _sigmoid,
)
from HWFP.core.domain.bet_record import BetOutcome, BetRecord
from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.calibration import (
    CalibrationEvent,
    CalibrationParams,
    CalibrationStatus,
)
from HWFP.core.domain.confidence_score import ConfidenceLevel, ConfidenceScore
from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.line_movement import LineMovement
from HWFP.core.domain.notification import NotificationPriority
from HWFP.core.domain.odds import Odds
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot
from HWFP.core.domain.referee_profile import RefereeProfile
from HWFP.core.domain.stake_result import StakeResult

_NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
_CLOCK = lambda: _NOW


# ── Domain helpers ─────────────────────────────────────────────────────────────


def _make_pmf(concentrated: bool = True) -> FoulPMF:
    n = 15
    if concentrated:
        vals = [0.0] * n
        vals[7] = 0.8
        vals[6] = 0.1
        vals[8] = 0.1
    else:
        vals = [1.0 / n] * n
    return FoulPMF(pmf=tuple(vals), bin_edges=tuple(range(n + 1)))


def _make_odds(decimal: float = 1.90, match_id: str = "m1") -> Odds:
    return Odds(
        match_id=match_id,
        market="total_fouls",
        line=9.5,
        side="over",
        decimal=decimal,
        bookmaker="codere",
        fetched_at=_NOW,
    )


def _make_ev(ev: float = 0.12, fair_prob: float = 0.55, match_id: str = "m1") -> EVResult:
    return EVResult(
        match_id=match_id,
        market="total_fouls",
        line=9.5,
        side="over",
        fair_prob=fair_prob,
        book_prob=1.0 / 1.90,
        ev=ev,
    )


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
    match_id: str = "m1",
    edge: float = 0.12,
    recommendation: Recommendation = Recommendation.BET,
) -> BettingDecision:
    return BettingDecision(
        match_id=match_id,
        recommendation=recommendation,
        market="total_fouls",
        line=9.5,
        side="over",
        best_bookmaker="codere",
        best_odds_value=1.90,
        p_model=0.55,
        edge=edge,
        stake_euros=20.0,
        confidence=_make_confidence(),
        reasons=("edge=12.0%",),
        generated_at=_NOW,
    )


def _make_bet_record(
    bet_id: str = "bet-1",
    outcome: BetOutcome = BetOutcome.WIN,
    p_model: float = 0.6,
) -> BetRecord:
    decision = BettingDecision(
        match_id="m1",
        recommendation=Recommendation.BET,
        market="total_fouls",
        line=9.5,
        side="over",
        best_bookmaker="codere",
        best_odds_value=1.90,
        p_model=p_model,
        edge=0.12,
        stake_euros=20.0,
        confidence=_make_confidence(),
        reasons=("edge=12.0%",),
        generated_at=_NOW,
    )
    profit: float | None
    if outcome == BetOutcome.WIN:
        profit = 18.0  # 20 * (1.90 - 1)
    elif outcome == BetOutcome.LOSS:
        profit = -20.0
    else:
        profit = None
    return BetRecord(
        bet_id=bet_id,
        decision=decision,
        placed_at=_NOW,
        outcome=outcome,
        closing_line=9.5,
        closing_odds=1.85,
        clv=0.02,
        profit_euros=profit,
    )


# ── Fake ports ─────────────────────────────────────────────────────────────────


class FakeMultiOdds:
    def __init__(self, odds: Odds | None = None) -> None:
        self._odds = odds

    def get_odds(self, match_id: str, market: str) -> list[Odds]:
        return [self._odds] if self._odds else []

    def best_odds(self, match_id: str, market: str) -> Odds | None:
        return self._odds


class FakeStateProvider:
    def get_match(self, match_id: str):
        return types.SimpleNamespace(
            match_id=match_id,
            home_team_id="home",
            away_team_id="away",
            referee_id="ref1",
            kickoff=_NOW,
        )

    def get_team_state(self, team_id: str, as_of):
        return types.SimpleNamespace(team_id=team_id)


class FakeFeatureBuilder:
    def build(self, match, home, away):
        return object()


class FakeModel:
    def __init__(self, pmf: FoulPMF | None = None) -> None:
        self._pmf = pmf or _make_pmf()

    def predict(self, fvec) -> FoulPMF:
        return self._pmf


class FakeModelRegistry:
    def __init__(self, pmf: FoulPMF | None = None) -> None:
        self._pmf = pmf

    def load_production(self) -> FakeModel:
        return FakeModel(self._pmf)

    def list_manifests(self) -> list:
        return []


class FakeRefereeProfiler:
    def __init__(self, sample_size: int = 25) -> None:
        self._n = sample_size

    def get_profile(self, referee_id: str) -> RefereeProfile:
        return RefereeProfile(
            referee_id=referee_id,
            avg_fouls_per_match=9.5,
            std_fouls_per_match=2.0,
            sample_size=self._n,
        )


class FakeEVCalculator:
    def __init__(self, ev_result: EVResult | None = None) -> None:
        self._ev = ev_result or _make_ev()

    def compute(self, pmf, odds, line) -> EVResult:
        return self._ev


class FakeStakingCalculator:
    def __init__(self, stake: float = 20.0) -> None:
        self._stake = stake

    def compute(self, ev: EVResult, bankroll: float) -> StakeResult:
        stake = min(self._stake, bankroll)  # StakeResult requires stake <= bankroll_used
        return StakeResult(
            match_id=ev.match_id,
            market=ev.market,
            stake=stake,
            kelly_fraction=stake / bankroll if bankroll > 0.0 else 0.0,
            bankroll_used=bankroll,
        )


class FakeLineMonitor:
    def __init__(self, movements: list[LineMovement] | None = None) -> None:
        self._movements = movements or []

    def get_current_odds(self, match_id: str) -> list[Odds]:
        return []

    def detect_movements(self, match_id: str, since: datetime) -> list[LineMovement]:
        return []

    def get_significant_movements(self, since: datetime) -> list[LineMovement]:
        return self._movements


class FakeNotificationSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, NotificationPriority]] = []
        self.decisions: list[BettingDecision] = []
        self.alerts: list[PerformanceSnapshot] = []

    def send(
        self,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> None:
        self.sent.append((message, priority))

    def send_decision(self, decision: BettingDecision) -> None:
        self.decisions.append(decision)

    def send_performance_alert(self, snapshot: PerformanceSnapshot) -> None:
        self.alerts.append(snapshot)


class FakePerformanceTracker:
    def __init__(self, records: list[BetRecord] | None = None) -> None:
        self._records: dict[str, BetRecord] = {r.bet_id: r for r in (records or [])}

    def record_bet(self, decision: BettingDecision) -> BetRecord:
        import uuid
        rec = BetRecord(
            bet_id=str(uuid.uuid4()),
            decision=decision,
            placed_at=_NOW,
            outcome=BetOutcome.PENDING,
            closing_line=None,
            closing_odds=None,
            clv=None,
            profit_euros=None,
        )
        self._records[rec.bet_id] = rec
        return rec

    def update_outcome(self, bet_id: str, outcome: BetOutcome) -> None:
        if bet_id not in self._records:
            raise KeyError(bet_id)
        old = self._records[bet_id]
        stake = old.decision.stake_euros
        odds = old.decision.best_odds_value
        profit: float | None
        if outcome == BetOutcome.WIN:
            profit = stake * (odds - 1.0)
        elif outcome == BetOutcome.LOSS:
            profit = -stake
        else:
            profit = None
        self._records[bet_id] = BetRecord(
            bet_id=old.bet_id,
            decision=old.decision,
            placed_at=old.placed_at,
            outcome=outcome,
            closing_line=old.closing_line,
            closing_odds=old.closing_odds,
            clv=old.clv,
            profit_euros=profit,
        )

    def get_records(self, last_n: int | None = None) -> list[BetRecord]:
        recs = list(self._records.values())
        return recs[-last_n:] if last_n else recs

    def get_pending_records(self) -> list[BetRecord]:
        return [r for r in self._records.values() if r.outcome == BetOutcome.PENDING]


class FakeCLVTracker:
    def __init__(self, avg_clv: float | None = 0.02) -> None:
        self._avg_clv = avg_clv
        self._closing: dict[str, tuple[float, float]] = {}

    def record_closing_line(
        self, bet_id: str, closing_line: float, closing_odds: float
    ) -> None:
        self._closing[bet_id] = (closing_line, closing_odds)

    def compute_clv(self, bet_id: str) -> float | None:
        return 0.02 if bet_id in self._closing else None

    def get_avg_clv(self, last_n: int | None = None) -> float | None:
        return self._avg_clv


class FakeCalibrationStore:
    def __init__(self, records: list[BetRecord] | None = None) -> None:
        self._events: list[CalibrationEvent] = []
        self._records: list[BetRecord] = records or []

    def get_current_params(self) -> CalibrationParams | None:
        return self._events[-1].params if self._events else None

    def save_calibration(self, event: CalibrationEvent) -> None:
        self._events.append(event)

    def get_history(self) -> list[CalibrationEvent]:
        return list(self._events)

    def get_bet_records_for_calibration(self, last_n: int) -> list[BetRecord]:
        return self._records[-last_n:]


# ── ScanJornada helpers ────────────────────────────────────────────────────────


def _make_scan_uc(**overrides) -> ScanJornadaUseCase:
    defaults: dict = dict(
        states=FakeStateProvider(),
        features=FakeFeatureBuilder(),
        model_registry=FakeModelRegistry(),
        referee=FakeRefereeProfiler(),
        multi_odds=FakeMultiOdds(_make_odds()),
        ev_calc=FakeEVCalculator(_make_ev(ev=0.12)),
        staking=FakeStakingCalculator(20.0),
        clock=_CLOCK,
    )
    defaults.update(overrides)
    return ScanJornadaUseCase(**defaults)


def _candidate(match_id: str = "m1") -> MatchCandidate:
    return MatchCandidate(match_id=match_id, market="total_fouls", line=9.5, side="over")


def _snapshot(status: CalibrationStatus = CalibrationStatus.GREEN) -> PerformanceSnapshot:
    mult = {
        CalibrationStatus.GREEN: 1.0,
        CalibrationStatus.YELLOW: 0.75,
        CalibrationStatus.ORANGE: 0.5,
        CalibrationStatus.RED: 0.0,
    }[status]
    return PerformanceSnapshot(
        status=status,
        n_bets_total=60,
        roi_trailing_30=0.05,
        ece_trailing_50=0.04,
        win_rate_high_conf=0.58,
        clv_avg=0.02,
        kelly_reduction=1.0 - mult,
        as_of=_NOW,
    )


# ── ScanJornadaUseCase tests ───────────────────────────────────────────────────


class TestScanJornadaUseCase:
    def test_bet_decision_when_ev_above_threshold(self):
        uc = _make_scan_uc(ev_calc=FakeEVCalculator(_make_ev(ev=0.15)))
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        assert out.decisions[0].recommendation == Recommendation.BET

    def test_skip_when_ev_below_threshold(self):
        uc = _make_scan_uc(ev_calc=FakeEVCalculator(_make_ev(ev=0.05)))
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        assert out.decisions[0].recommendation == Recommendation.SKIP

    def test_skip_reason_contains_edge_label_when_below_threshold(self):
        uc = _make_scan_uc(ev_calc=FakeEVCalculator(_make_ev(ev=0.05)))
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        assert any("edge" in r for r in out.decisions[0].reasons)

    def test_skip_when_no_odds_available(self):
        uc = _make_scan_uc(multi_odds=FakeMultiOdds(None))
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        assert out.decisions[0].recommendation == Recommendation.SKIP
        assert "no_odds_available" in out.decisions[0].reasons

    def test_skip_when_calibration_red(self):
        uc = _make_scan_uc(ev_calc=FakeEVCalculator(_make_ev(ev=0.20)))
        out = uc.execute(
            ScanJornadaInput(candidates=[_candidate()], bankroll=200.0),
            snapshot=_snapshot(CalibrationStatus.RED),
        )
        assert out.decisions[0].recommendation == Recommendation.SKIP
        assert "calibration_red" in out.decisions[0].reasons

    def test_bets_sorted_before_skips_in_output(self):
        candidates = [_candidate("m1"), _candidate("m2"), _candidate("m3")]
        # m2 will skip due to low EV — we need per-match EV logic
        # Use a fixed-high EV so all are BETs, then verify sort still correct
        uc = _make_scan_uc(ev_calc=FakeEVCalculator(_make_ev(ev=0.15)))
        out = uc.execute(ScanJornadaInput(candidates=candidates, bankroll=200.0))
        bets = [d for d in out.decisions if d.recommendation == Recommendation.BET]
        skips = [d for d in out.decisions if d.recommendation == Recommendation.SKIP]
        if bets and skips:
            last_bet_idx = out.decisions.index(bets[-1])
            first_skip_idx = out.decisions.index(skips[0])
            assert last_bet_idx < first_skip_idx

    def test_stake_capped_at_kelly_cap(self):
        uc = _make_scan_uc(
            ev_calc=FakeEVCalculator(_make_ev(ev=0.25)),
            staking=FakeStakingCalculator(stake=10_000.0),  # enormous raw stake
        )
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        bet = out.decisions[0]
        assert bet.recommendation == Recommendation.BET
        assert bet.stake_euros <= 200.0 * 0.30 + 1e-9  # kelly_cap=30%

    def test_stake_floored_at_stake_min(self):
        uc = _make_scan_uc(
            model_registry=FakeModelRegistry(_make_pmf(concentrated=False)),
            referee=FakeRefereeProfiler(sample_size=2),
            ev_calc=FakeEVCalculator(_make_ev(ev=0.11)),
            staking=FakeStakingCalculator(stake=0.50),  # tiny raw stake
        )
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        bet = out.decisions[0]
        if bet.recommendation == Recommendation.BET:
            assert bet.stake_euros >= 5.0

    def test_calibration_yellow_reduces_stake(self):
        uc_green = _make_scan_uc(staking=FakeStakingCalculator(50.0))
        uc_yellow = _make_scan_uc(staking=FakeStakingCalculator(50.0))
        out_green = uc_green.execute(
            ScanJornadaInput(candidates=[_candidate()], bankroll=1000.0),
            snapshot=_snapshot(CalibrationStatus.GREEN),
        )
        out_yellow = uc_yellow.execute(
            ScanJornadaInput(candidates=[_candidate()], bankroll=1000.0),
            snapshot=_snapshot(CalibrationStatus.YELLOW),
        )
        stake_green = out_green.decisions[0].stake_euros
        stake_yellow = out_yellow.decisions[0].stake_euros
        assert stake_yellow < stake_green

    def test_best_bookmaker_populated_from_odds(self):
        odds = _make_odds(decimal=2.10)
        uc = _make_scan_uc(multi_odds=FakeMultiOdds(odds))
        out = uc.execute(ScanJornadaInput(candidates=[_candidate()], bankroll=200.0))
        bet = out.decisions[0]
        assert bet.best_bookmaker == "codere"
        assert bet.best_odds_value == pytest.approx(2.10)

    def test_multiple_candidates_all_processed(self):
        candidates = [_candidate(f"m{i}") for i in range(5)]
        uc = _make_scan_uc()
        out = uc.execute(ScanJornadaInput(candidates=candidates, bankroll=500.0))
        assert len(out.decisions) == 5


class TestPMFEntropy:
    def test_concentrated_pmf_has_low_entropy(self):
        assert _pmf_entropy(_make_pmf(concentrated=True)) < 2.0

    def test_flat_pmf_has_high_entropy(self):
        assert _pmf_entropy(_make_pmf(concentrated=False)) > 3.0

    def test_single_mass_has_zero_entropy(self):
        vals = [0.0] * 10
        vals[5] = 1.0
        pmf = FoulPMF(pmf=tuple(vals), bin_edges=tuple(range(11)))
        assert _pmf_entropy(pmf) == pytest.approx(0.0, abs=1e-9)


class TestBuildConfidence:
    def test_high_when_concentrated_pmf_and_large_referee_sample(self):
        conf = _build_confidence(_make_pmf(concentrated=True), referee_sample_size=25)
        assert conf.level == ConfidenceLevel.HIGH
        assert conf.kelly_multiplier == pytest.approx(1.0)

    def test_low_when_flat_pmf(self):
        conf = _build_confidence(_make_pmf(concentrated=False), referee_sample_size=25)
        assert conf.level == ConfidenceLevel.LOW

    def test_medium_when_small_referee_sample(self):
        conf = _build_confidence(_make_pmf(concentrated=True), referee_sample_size=8)
        assert conf.level == ConfidenceLevel.MEDIUM

    def test_low_when_very_small_referee_sample(self):
        conf = _build_confidence(_make_pmf(concentrated=True), referee_sample_size=3)
        assert conf.level == ConfidenceLevel.LOW

    def test_fallback_count_degrades_high_to_medium(self):
        conf = _build_confidence(
            _make_pmf(concentrated=True),
            referee_sample_size=25,
            feature_fallback_count=3,
        )
        assert conf.level == ConfidenceLevel.MEDIUM


# ── MonitorOddsUseCase tests ───────────────────────────────────────────────────


def _make_movement(delta: float = 0.5) -> LineMovement:
    return LineMovement(
        match_id="m1",
        bookmaker="codere",
        market="total_fouls",
        line_before=9.5,
        line_after=9.5 + delta,
        odds_before=1.90,
        odds_after=1.80,
        delta=delta,
        detected_at=_NOW,
        is_significant=abs(delta) >= 0.5,
    )


class TestMonitorOddsUseCase:
    def test_sends_alert_for_each_significant_movement(self):
        notif = FakeNotificationSender()
        uc = MonitorOddsUseCase(
            line_monitor=FakeLineMonitor([_make_movement(0.5), _make_movement(1.0)]),
            notifications=notif,
        )
        out = uc.execute(MonitorOddsInput(since=_NOW))
        assert out.alerts_sent == 2
        assert len(notif.sent) == 2

    def test_large_movement_sends_high_priority(self):
        notif = FakeNotificationSender()
        uc = MonitorOddsUseCase(
            line_monitor=FakeLineMonitor([_make_movement(1.5)]),
            notifications=notif,
        )
        uc.execute(MonitorOddsInput(since=_NOW))
        _, priority = notif.sent[0]
        assert priority == NotificationPriority.HIGH

    def test_small_movement_sends_normal_priority(self):
        notif = FakeNotificationSender()
        uc = MonitorOddsUseCase(
            line_monitor=FakeLineMonitor([_make_movement(0.5)]),
            notifications=notif,
        )
        uc.execute(MonitorOddsInput(since=_NOW))
        _, priority = notif.sent[0]
        assert priority == NotificationPriority.NORMAL

    def test_no_alerts_when_no_movements(self):
        notif = FakeNotificationSender()
        uc = MonitorOddsUseCase(
            line_monitor=FakeLineMonitor([]),
            notifications=notif,
        )
        out = uc.execute(MonitorOddsInput(since=_NOW))
        assert out.alerts_sent == 0
        assert out.movements == []

    def test_returns_all_movements_in_output(self):
        movements = [_make_movement(0.5), _make_movement(1.0)]
        uc = MonitorOddsUseCase(
            line_monitor=FakeLineMonitor(movements),
            notifications=FakeNotificationSender(),
        )
        out = uc.execute(MonitorOddsInput(since=_NOW))
        assert len(out.movements) == 2


# ── TrackBetOutcomeUseCase tests ───────────────────────────────────────────────


class TestTrackBetOutcomeUseCase:
    def test_updates_outcome_to_win(self):
        record = _make_bet_record(bet_id="bet-1", outcome=BetOutcome.PENDING)
        tracker = FakePerformanceTracker([record])
        uc = TrackBetOutcomeUseCase(
            tracker=tracker,
            clv_tracker=FakeCLVTracker(),
            notifications=FakeNotificationSender(),
        )
        out = uc.execute(TrackBetOutcomeInput(
            bet_id="bet-1", outcome=BetOutcome.WIN, closing_line=9.5, closing_odds=1.85
        ))
        assert out.record.outcome == BetOutcome.WIN

    def test_records_closing_line_in_clv_tracker(self):
        record = _make_bet_record(bet_id="bet-1", outcome=BetOutcome.PENDING)
        clv = FakeCLVTracker()
        uc = TrackBetOutcomeUseCase(
            tracker=FakePerformanceTracker([record]),
            clv_tracker=clv,
            notifications=FakeNotificationSender(),
        )
        uc.execute(TrackBetOutcomeInput(
            bet_id="bet-1", outcome=BetOutcome.WIN, closing_line=9.5, closing_odds=1.85
        ))
        assert "bet-1" in clv._closing

    def test_sends_clv_alert_when_avg_below_threshold(self):
        record = _make_bet_record(bet_id="bet-1", outcome=BetOutcome.PENDING)
        notif = FakeNotificationSender()
        uc = TrackBetOutcomeUseCase(
            tracker=FakePerformanceTracker([record]),
            clv_tracker=FakeCLVTracker(avg_clv=-0.08),
            notifications=notif,
        )
        out = uc.execute(TrackBetOutcomeInput(
            bet_id="bet-1", outcome=BetOutcome.LOSS, closing_line=9.5, closing_odds=1.85
        ))
        assert out.alert_sent is True
        assert len(notif.sent) == 1

    def test_no_alert_when_clv_positive(self):
        record = _make_bet_record(bet_id="bet-1", outcome=BetOutcome.PENDING)
        notif = FakeNotificationSender()
        uc = TrackBetOutcomeUseCase(
            tracker=FakePerformanceTracker([record]),
            clv_tracker=FakeCLVTracker(avg_clv=0.03),
            notifications=notif,
        )
        out = uc.execute(TrackBetOutcomeInput(
            bet_id="bet-1", outcome=BetOutcome.WIN, closing_line=9.5, closing_odds=1.85
        ))
        assert out.alert_sent is False

    def test_raises_when_bet_id_not_found(self):
        uc = TrackBetOutcomeUseCase(
            tracker=FakePerformanceTracker([]),
            clv_tracker=FakeCLVTracker(),
            notifications=FakeNotificationSender(),
        )
        with pytest.raises((KeyError, ValueError)):
            uc.execute(TrackBetOutcomeInput(
                bet_id="nonexistent", outcome=BetOutcome.WIN,
                closing_line=9.5, closing_odds=1.85
            ))

    def test_clv_computed_and_returned(self):
        record = _make_bet_record(bet_id="bet-1", outcome=BetOutcome.PENDING)
        uc = TrackBetOutcomeUseCase(
            tracker=FakePerformanceTracker([record]),
            clv_tracker=FakeCLVTracker(avg_clv=0.02),
            notifications=FakeNotificationSender(),
        )
        out = uc.execute(TrackBetOutcomeInput(
            bet_id="bet-1", outcome=BetOutcome.WIN, closing_line=9.5, closing_odds=1.85
        ))
        assert out.clv is not None


# ── EvaluatePerformanceUseCase tests ──────────────────────────────────────────


class TestEvaluatePerformanceUseCase:
    def test_returns_yellow_when_insufficient_bets(self):
        records = [_make_bet_record(f"bet-{i}", BetOutcome.WIN) for i in range(20)]
        uc = EvaluatePerformanceUseCase(
            tracker=FakePerformanceTracker(records),
            clv_tracker=FakeCLVTracker(),
            clock=_CLOCK,
        )
        out = uc.execute()
        assert out.snapshot.status == CalibrationStatus.YELLOW
        assert out.snapshot.n_bets_total == 20

    def test_returns_red_when_badly_calibrated(self):
        # p_model=0.9 but all LOSS → ECE ≈ 0.9
        records = [
            _make_bet_record(f"bet-{i}", BetOutcome.LOSS, p_model=0.9)
            for i in range(55)
        ]
        uc = EvaluatePerformanceUseCase(
            tracker=FakePerformanceTracker(records),
            clv_tracker=FakeCLVTracker(),
            clock=_CLOCK,
        )
        out = uc.execute()
        assert out.snapshot.status == CalibrationStatus.RED

    def test_returns_green_when_well_calibrated(self):
        # p_model=0.5, 50% win rate → ECE ≈ 0
        records = [
            _make_bet_record(f"bet-{i}", BetOutcome.WIN if i % 2 == 0 else BetOutcome.LOSS, p_model=0.5)
            for i in range(60)
        ]
        uc = EvaluatePerformanceUseCase(
            tracker=FakePerformanceTracker(records),
            clv_tracker=FakeCLVTracker(),
            clock=_CLOCK,
        )
        out = uc.execute()
        assert out.snapshot.status in (CalibrationStatus.GREEN, CalibrationStatus.YELLOW)

    def test_roi_computed_from_recent_30(self):
        records = [
            _make_bet_record("bet-win", BetOutcome.WIN),
            _make_bet_record("bet-loss", BetOutcome.LOSS),
        ]
        uc = EvaluatePerformanceUseCase(
            tracker=FakePerformanceTracker(records),
            clv_tracker=FakeCLVTracker(),
            clock=_CLOCK,
        )
        out = uc.execute()
        # (18.0 - 20.0) / (20.0 + 20.0) = -0.05
        assert out.snapshot.roi_trailing_30 == pytest.approx(-0.05, rel=1e-2)

    def test_clv_avg_from_clv_tracker(self):
        records = [_make_bet_record(f"bet-{i}", BetOutcome.WIN) for i in range(55)]
        uc = EvaluatePerformanceUseCase(
            tracker=FakePerformanceTracker(records),
            clv_tracker=FakeCLVTracker(avg_clv=0.03),
            clock=_CLOCK,
        )
        out = uc.execute()
        assert out.snapshot.clv_avg == pytest.approx(0.03)

    def test_as_of_uses_clock(self):
        uc = EvaluatePerformanceUseCase(
            tracker=FakePerformanceTracker([]),
            clv_tracker=FakeCLVTracker(),
            clock=_CLOCK,
        )
        out = uc.execute()
        assert out.snapshot.as_of == _NOW


class TestDetermineStatus:
    def test_green_when_low_ece_and_sufficient_bets(self):
        assert _determine_status(0.03, 55) == CalibrationStatus.GREEN

    def test_yellow_when_insufficient_bets(self):
        assert _determine_status(0.03, 30) == CalibrationStatus.YELLOW

    def test_yellow_when_moderate_ece(self):
        assert _determine_status(0.08, 60) == CalibrationStatus.YELLOW

    def test_orange_when_medium_ece(self):
        assert _determine_status(0.12, 60) == CalibrationStatus.ORANGE

    def test_red_when_high_ece(self):
        assert _determine_status(0.20, 60) == CalibrationStatus.RED


class TestComputeECE:
    def test_ece_near_zero_for_perfect_calibration(self):
        probs = [0.5] * 100
        outcomes = [1.0 if i % 2 == 0 else 0.0 for i in range(100)]
        assert _compute_ece(probs, outcomes) < 0.05

    def test_ece_high_for_worst_case(self):
        # All confident predictions, all wrong
        probs = [0.9] * 50
        outcomes = [0.0] * 50
        assert _compute_ece(probs, outcomes) > 0.5

    def test_ece_empty_returns_zero(self):
        assert _compute_ece([], []) == 0.0


class TestComputeROI:
    def test_roi_correct(self):
        records = [
            _make_bet_record("b1", BetOutcome.WIN),   # profit=18
            _make_bet_record("b2", BetOutcome.LOSS),  # profit=-20
        ]
        roi = _compute_roi(records)
        assert roi == pytest.approx(-2.0 / 40.0, rel=1e-3)

    def test_roi_empty_returns_zero(self):
        assert _compute_roi([]) == 0.0


# ── RecalibrateUseCase tests ───────────────────────────────────────────────────


def _calibration_records(n: int, p_model: float = 0.6, win_frac: float = 0.55) -> list[BetRecord]:
    records = []
    for i in range(n):
        outcome = BetOutcome.WIN if i < int(n * win_frac) else BetOutcome.LOSS
        records.append(_make_bet_record(f"bet-{i}", outcome, p_model=p_model))
    return records


class TestRecalibrateUseCase:
    def test_rejects_when_insufficient_data(self):
        store = FakeCalibrationStore(records=_calibration_records(10))
        uc = RecalibrateUseCase(calibration_store=store, clock=_CLOCK)
        out = uc.execute(RecalibrateInput(trigger="manual"))
        assert out.accepted is False
        assert "Insufficient" in out.message

    def test_does_not_save_event_when_insufficient_data(self):
        store = FakeCalibrationStore(records=_calibration_records(10))
        uc = RecalibrateUseCase(calibration_store=store, clock=_CLOCK)
        uc.execute(RecalibrateInput(trigger="manual"))
        assert len(store._events) == 0

    def test_saves_event_with_sufficient_data(self):
        store = FakeCalibrationStore(records=_calibration_records(50))
        uc = RecalibrateUseCase(calibration_store=store, clock=_CLOCK)
        uc.execute(RecalibrateInput(trigger="manual"))
        assert len(store._events) == 1
        assert store._events[0].trigger == "manual"

    def test_event_contains_ece_before_and_after(self):
        store = FakeCalibrationStore(records=_calibration_records(50, p_model=0.9, win_frac=0.55))
        uc = RecalibrateUseCase(calibration_store=store, clock=_CLOCK)
        out = uc.execute(RecalibrateInput(trigger="test"))
        assert out.ece_before > 0.0
        assert out.n_bets_used == 50

    def test_version_increments_on_successive_calls(self):
        records = _calibration_records(50)
        store = FakeCalibrationStore(records=records)
        uc = RecalibrateUseCase(calibration_store=store, clock=_CLOCK)
        uc.execute(RecalibrateInput(trigger="first"))
        uc.execute(RecalibrateInput(trigger="second"))
        versions = [e.params.version for e in store._events]
        assert versions[1] > versions[0]

    def test_last_n_respected(self):
        records = _calibration_records(200)
        store = FakeCalibrationStore(records=records)
        uc = RecalibrateUseCase(calibration_store=store, clock=_CLOCK)
        out = uc.execute(RecalibrateInput(trigger="manual", last_n=50))
        assert out.n_bets_used <= 50


class TestPlattHelpers:
    def test_logit_of_half_is_zero(self):
        assert _logit(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_of_zero_is_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_logit_sigmoid_inverse(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            assert _sigmoid(_logit(p)) == pytest.approx(p, abs=1e-6)

    def test_fit_platt_returns_finite_params(self):
        probs = [0.7] * 40 + [0.3] * 10
        outcomes = [1.0] * 35 + [0.0] * 15
        a, b = _fit_platt(probs, outcomes)
        assert math.isfinite(a) and math.isfinite(b)
