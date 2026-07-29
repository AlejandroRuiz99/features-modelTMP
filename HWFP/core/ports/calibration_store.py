"""CalibrationStore port — persist and retrieve calibration state."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from HWFP.core.domain.bet_record import BetRecord
from HWFP.core.domain.calibration import CalibrationEvent, CalibrationParams


@runtime_checkable
class CalibrationStore(Protocol):
    """Persist calibration parameters and history."""

    def get_current_params(self) -> CalibrationParams | None: ...
    def save_calibration(self, event: CalibrationEvent) -> None: ...
    def get_history(self) -> list[CalibrationEvent]: ...
    def get_bet_records_for_calibration(self, last_n: int) -> list[BetRecord]: ...
