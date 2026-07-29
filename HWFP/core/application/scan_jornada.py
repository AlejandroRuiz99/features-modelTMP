"""ScanJornadaUseCase — scan all match candidates for EV+ betting opportunities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.calibration import CalibrationStatus
from HWFP.core.domain.confidence_score import ConfidenceLevel, ConfidenceScore
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot
from HWFP.core.ports.ev_calculator import EVCalculator
from HWFP.core.ports.feature_builder import FeatureBuilder
from HWFP.core.ports.model_registry import ModelRegistry
from HWFP.core.ports.multi_odds_provider import MultiOddsProvider
from HWFP.core.ports.referee_profiler import RefereeProfiler
from HWFP.core.ports.staking_calculator import StakingCalculator
from HWFP.core.ports.state_provider import StateProvider

_LEVEL_ORDER = {ConfidenceLevel.HIGH: 2, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 0}
_KELLY_MULT = {ConfidenceLevel.HIGH: 1.0, ConfidenceLevel.MEDIUM: 0.6, ConfidenceLevel.LOW: 0.25}
_CALIB_MULT = {
    CalibrationStatus.GREEN: 1.0,
    CalibrationStatus.YELLOW: 0.75,
    CalibrationStatus.ORANGE: 0.5,
    CalibrationStatus.RED: 0.0,
}

# Entropy thresholds (bits): H < LOW_MAX → HIGH, H < MED_MAX → MEDIUM, else LOW
_ENTROPY_HIGH_MAX = 2.0
_ENTROPY_MED_MAX = 3.0

# Referee sample thresholds
_REF_HIGH_MIN = 20
_REF_MED_MIN = 5


@dataclass(frozen=True)
class MatchCandidate:
    match_id: str
    market: str
    line: float
    side: str


@dataclass(frozen=True)
class ScanJornadaInput:
    candidates: list[MatchCandidate]
    bankroll: float


@dataclass(frozen=True)
class ScanJornadaOutput:
    decisions: list[BettingDecision]


class ScanJornadaUseCase:
    def __init__(
        self,
        states: StateProvider,
        features: FeatureBuilder,
        model_registry: ModelRegistry,
        referee: RefereeProfiler,
        multi_odds: MultiOddsProvider,
        ev_calc: EVCalculator,
        staking: StakingCalculator,
        clock: Callable[[], datetime],
        edge_min: float = 0.10,
        kelly_cap: float = 0.30,
        stake_min: float = 5.0,
    ) -> None:
        self._states = states
        self._features = features
        self._model_registry = model_registry
        self._referee = referee
        self._multi_odds = multi_odds
        self._ev_calc = ev_calc
        self._staking = staking
        self._clock = clock
        self._edge_min = edge_min
        self._kelly_cap = kelly_cap
        self._stake_min = stake_min

    def execute(
        self,
        inp: ScanJornadaInput,
        snapshot: PerformanceSnapshot | None = None,
    ) -> ScanJornadaOutput:
        calib_status = snapshot.status if snapshot else CalibrationStatus.GREEN
        calib_mult = _CALIB_MULT[calib_status]
        model = self._model_registry.load_production()

        decisions: list[BettingDecision] = []
        for candidate in inp.candidates:
            decisions.append(
                self._evaluate(candidate, inp.bankroll, model, calib_status, calib_mult)
            )

        bets = sorted(
            (d for d in decisions if d.recommendation == Recommendation.BET),
            key=lambda d: d.edge,
            reverse=True,
        )
        skips = [d for d in decisions if d.recommendation == Recommendation.SKIP]
        return ScanJornadaOutput(decisions=list(bets) + skips)

    def _evaluate(
        self,
        candidate: MatchCandidate,
        bankroll: float,
        model,
        calib_status: CalibrationStatus,
        calib_mult: float,
    ) -> BettingDecision:
        now = self._clock()

        best = self._multi_odds.best_odds(candidate.match_id, candidate.market)
        if best is None:
            return _skip(candidate, "no_odds_available", now)

        if calib_mult == 0.0:
            return _skip(candidate, "calibration_red", now)

        try:
            match = self._states.get_match(candidate.match_id)
            home = self._states.get_team_state(match.home_team_id, match.kickoff)
            away = self._states.get_team_state(match.away_team_id, match.kickoff)
            ref_profile = self._referee.get_profile(match.referee_id)
            fvec = self._features.build(match, home, away)
            pmf = model.predict(fvec)
        except Exception as exc:
            return _skip(candidate, f"error:{type(exc).__name__}", now)

        ev = self._ev_calc.compute(pmf, best, candidate.line)

        if ev.ev < self._edge_min:
            return _skip(candidate, f"edge_below_{self._edge_min:.0%}", now)

        confidence = _build_confidence(pmf, ref_profile.sample_size)
        base_stake = self._staking.compute(ev, bankroll).stake
        final_stake = base_stake * confidence.kelly_multiplier * calib_mult
        final_stake = max(self._stake_min, min(final_stake, self._kelly_cap * bankroll))

        return BettingDecision(
            match_id=candidate.match_id,
            recommendation=Recommendation.BET,
            market=candidate.market,
            line=candidate.line,
            side=candidate.side,
            best_bookmaker=best.bookmaker,
            best_odds_value=best.decimal,
            p_model=ev.fair_prob,
            edge=ev.ev,
            stake_euros=final_stake,
            confidence=confidence,
            reasons=tuple(_build_reasons(ev, confidence, calib_status)),
            generated_at=now,
        )


# ── Pure helpers ───────────────────────────────────────────────────────────────


def _pmf_entropy(pmf: FoulPMF) -> float:
    return -sum(p * math.log2(p) for p in pmf.pmf if p > 0.0)


def _entropy_level(entropy: float) -> ConfidenceLevel:
    if entropy < _ENTROPY_HIGH_MAX:
        return ConfidenceLevel.HIGH
    if entropy < _ENTROPY_MED_MAX:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _referee_level(sample_size: int) -> ConfidenceLevel:
    if sample_size >= _REF_HIGH_MIN:
        return ConfidenceLevel.HIGH
    if sample_size >= _REF_MED_MIN:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _min_level(a: ConfidenceLevel, b: ConfidenceLevel) -> ConfidenceLevel:
    return a if _LEVEL_ORDER[a] <= _LEVEL_ORDER[b] else b


def _build_confidence(
    pmf: FoulPMF,
    referee_sample_size: int,
    feature_fallback_count: int = 0,
) -> ConfidenceScore:
    entropy = _pmf_entropy(pmf)
    level = _min_level(_entropy_level(entropy), _referee_level(referee_sample_size))
    if feature_fallback_count >= 3 and level == ConfidenceLevel.HIGH:
        level = ConfidenceLevel.MEDIUM
    return ConfidenceScore(
        pmf_entropy=entropy,
        referee_sample_size=referee_sample_size,
        feature_fallback_count=feature_fallback_count,
        kelly_multiplier=_KELLY_MULT[level],
        level=level,
    )


def _skip(candidate: MatchCandidate, reason: str, now: datetime) -> BettingDecision:
    return BettingDecision(
        match_id=candidate.match_id,
        recommendation=Recommendation.SKIP,
        market=candidate.market,
        line=candidate.line,
        side=candidate.side,
        best_bookmaker="",
        best_odds_value=0.0,
        p_model=0.0,
        edge=0.0,
        stake_euros=0.0,
        confidence=ConfidenceScore(
            pmf_entropy=0.0,
            referee_sample_size=0,
            feature_fallback_count=0,
            kelly_multiplier=0.0,
            level=ConfidenceLevel.LOW,
        ),
        reasons=(reason,),
        generated_at=now,
    )


def _build_reasons(ev, confidence: ConfidenceScore, calib_status: CalibrationStatus) -> list[str]:
    return [
        f"edge={ev.ev:.1%}",
        f"p_model={ev.fair_prob:.1%}",
        f"confidence={confidence.level.value}",
        f"calibration={calib_status.value}",
    ]
