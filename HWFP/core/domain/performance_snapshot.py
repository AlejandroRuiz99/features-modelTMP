from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from HWFP.core.domain.calibration import CalibrationStatus


@dataclass(frozen=True)
class PerformanceSnapshot:
    status: CalibrationStatus
    n_bets_total: int
    roi_trailing_30: float
    ece_trailing_50: float
    win_rate_high_conf: float
    clv_avg: float
    kelly_reduction: float
    as_of: datetime

    def kelly_multiplier_for_status(self) -> float:
        mapping = {
            CalibrationStatus.GREEN: 1.0,
            CalibrationStatus.YELLOW: 0.75,
            CalibrationStatus.ORANGE: 0.5,
            CalibrationStatus.RED: 0.0,
        }
        return mapping[self.status]
