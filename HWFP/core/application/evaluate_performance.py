"""EvaluatePerformanceUseCase — compute PerformanceSnapshot from recent bet records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from HWFP.core.domain.bet_record import BetOutcome, BetRecord
from HWFP.core.domain.calibration import CalibrationStatus
from HWFP.core.domain.confidence_score import ConfidenceLevel
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot
from HWFP.core.ports.clv_tracker import CLVTracker
from HWFP.core.ports.performance_tracker import PerformanceTracker

_N_ECE = 50    # bets used for ECE calculation
_N_ROI = 30    # bets used for ROI calculation
_N_MIN_GREEN = 50  # minimum settled bets to reach GREEN status

_ECE_GREEN = 0.05
_ECE_YELLOW = 0.10
_ECE_ORANGE = 0.15


@dataclass(frozen=True)
class EvaluatePerformanceOutput:
    snapshot: PerformanceSnapshot


class EvaluatePerformanceUseCase:
    def __init__(
        self,
        tracker: PerformanceTracker,
        clv_tracker: CLVTracker,
        clock: Callable[[], datetime],
    ) -> None:
        self._tracker = tracker
        self._clv = clv_tracker
        self._clock = clock

    def execute(self) -> EvaluatePerformanceOutput:
        all_records = self._tracker.get_records()
        settled = [
            r for r in all_records
            if r.outcome not in (BetOutcome.PENDING, BetOutcome.VOID)
        ]

        n_total = len(settled)
        recent_ece = settled[-_N_ECE:]
        recent_roi = settled[-_N_ROI:]

        probs_ece = [r.decision.p_model for r in recent_ece]
        outcomes_ece = [1.0 if r.outcome == BetOutcome.WIN else 0.0 for r in recent_ece]
        ece = _compute_ece(probs_ece, outcomes_ece) if n_total > 0 else 0.0
        roi = _compute_roi(recent_roi)
        win_rate_hc = _win_rate_high_conf(settled)
        avg_clv = self._clv.get_avg_clv(last_n=20)

        status = _determine_status(ece, n_total)
        snapshot = PerformanceSnapshot(
            status=status,
            n_bets_total=n_total,
            roi_trailing_30=roi,
            ece_trailing_50=ece,
            win_rate_high_conf=win_rate_hc,
            clv_avg=avg_clv if avg_clv is not None else 0.0,
            kelly_reduction=1.0 - snapshot_kelly(status),
            as_of=self._clock(),
        )
        return EvaluatePerformanceOutput(snapshot=snapshot)


def snapshot_kelly(status: CalibrationStatus) -> float:
    return {
        CalibrationStatus.GREEN: 1.0,
        CalibrationStatus.YELLOW: 0.75,
        CalibrationStatus.ORANGE: 0.5,
        CalibrationStatus.RED: 0.0,
    }[status]


def _determine_status(ece: float, n_total: int) -> CalibrationStatus:
    if n_total < _N_MIN_GREEN:
        return CalibrationStatus.YELLOW
    if ece <= _ECE_GREEN:
        return CalibrationStatus.GREEN
    if ece <= _ECE_YELLOW:
        return CalibrationStatus.YELLOW
    if ece <= _ECE_ORANGE:
        return CalibrationStatus.ORANGE
    return CalibrationStatus.RED


def _compute_ece(probs: list[float], outcomes: list[float], n_bins: int = 10) -> float:
    n = len(probs)
    if n == 0:
        return 0.0
    ece = 0.0
    bin_size = 1.0 / n_bins
    for i in range(n_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        in_bin = [(p, y) for p, y in zip(probs, outcomes) if lo <= p < hi]
        if not in_bin:
            continue
        bin_conf = sum(p for p, _ in in_bin) / len(in_bin)
        bin_acc = sum(y for _, y in in_bin) / len(in_bin)
        ece += abs(bin_acc - bin_conf) * len(in_bin) / n
    return ece


def _compute_roi(records: list[BetRecord]) -> float:
    if not records:
        return 0.0
    total_profit = sum(r.profit_euros for r in records if r.profit_euros is not None)
    total_stake = sum(r.decision.stake_euros for r in records)
    return total_profit / total_stake if total_stake > 0.0 else 0.0


def _win_rate_high_conf(records: list[BetRecord]) -> float:
    high = [r for r in records if r.decision.confidence.level == ConfidenceLevel.HIGH]
    if not high:
        return 0.0
    return sum(1 for r in high if r.outcome == BetOutcome.WIN) / len(high)
