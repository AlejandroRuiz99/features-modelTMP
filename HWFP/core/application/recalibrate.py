"""RecalibrateUseCase — fit Platt scaling on recent settled bet records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from HWFP.core.domain.bet_record import BetOutcome
from HWFP.core.domain.calibration import CalibrationEvent, CalibrationParams
from HWFP.core.ports.calibration_store import CalibrationStore

_MIN_BETS = 30         # minimum settled bets required to attempt calibration
_MIN_IMPROVEMENT = 0.005  # ECE must improve by at least 0.5pp to be accepted


@dataclass(frozen=True)
class RecalibrateInput:
    trigger: str           # "manual", "scheduled", "alert"
    last_n: int = 100


@dataclass(frozen=True)
class RecalibrateOutput:
    accepted: bool
    ece_before: float
    ece_after: float
    n_bets_used: int
    message: str


class RecalibrateUseCase:
    def __init__(
        self,
        calibration_store: CalibrationStore,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = calibration_store
        self._clock = clock

    def execute(self, inp: RecalibrateInput) -> RecalibrateOutput:
        records = self._store.get_bet_records_for_calibration(inp.last_n)
        settled = [r for r in records if r.outcome in (BetOutcome.WIN, BetOutcome.LOSS)]

        if len(settled) < _MIN_BETS:
            return RecalibrateOutput(
                accepted=False,
                ece_before=0.0,
                ece_after=0.0,
                n_bets_used=len(settled),
                message=(
                    f"Insufficient data: {len(settled)}/{_MIN_BETS} settled bets required"
                ),
            )

        probs = [r.decision.p_model for r in settled]
        outcomes = [1.0 if r.outcome == BetOutcome.WIN else 0.0 for r in settled]

        ece_before = _compute_ece(probs, outcomes)
        a, b = _fit_platt(probs, outcomes)
        calibrated = [_sigmoid(a * _logit(p) + b) for p in probs]
        ece_after = _compute_ece(calibrated, outcomes)

        improvement = ece_before - ece_after
        accepted = improvement >= _MIN_IMPROVEMENT

        version = _next_version(self._store)
        params = CalibrationParams(
            a=a,
            b=b,
            n_bets_fitted=len(settled),
            ece_before=ece_before,
            ece_after=ece_after,
            fitted_at=self._clock(),
            version=version,
        )
        event = CalibrationEvent(
            params=params,
            trigger=inp.trigger,
            accepted=accepted,
            recorded_at=self._clock(),
        )
        self._store.save_calibration(event)

        msg = (
            f"Calibration {'accepted' if accepted else 'rejected'}: "
            f"ECE {ece_before:.4f} → {ece_after:.4f} "
            f"(Δ={improvement:+.4f}, n={len(settled)})"
        )
        return RecalibrateOutput(
            accepted=accepted,
            ece_before=ece_before,
            ece_after=ece_after,
            n_bets_used=len(settled),
            message=msg,
        )


# ── Pure helpers ───────────────────────────────────────────────────────────────


def _logit(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _fit_platt(probs: list[float], outcomes: list[float]) -> tuple[float, float]:
    try:
        import numpy as np
        from scipy.optimize import minimize

        f = np.array([_logit(p) for p in probs])
        y = np.array(outcomes)

        def neg_ll(params: np.ndarray) -> float:
            a, b = params
            p_cal = 1.0 / (1.0 + np.exp(-(a * f + b)))
            p_cal = np.clip(p_cal, 1e-9, 1.0 - 1e-9)
            return float(-np.sum(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal)))

        result = minimize(neg_ll, [1.0, 0.0], method="BFGS")
        return float(result.x[0]), float(result.x[1])
    except ImportError:
        return 1.0, 0.0


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


def _next_version(store: CalibrationStore) -> int:
    history = store.get_history()
    if not history:
        return 1
    return max(e.params.version for e in history) + 1
