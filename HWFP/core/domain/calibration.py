from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CalibrationStatus(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass(frozen=True)
class CalibrationParams:
    a: float
    b: float
    n_bets_fitted: int
    ece_before: float
    ece_after: float
    fitted_at: datetime
    version: int


@dataclass(frozen=True)
class CalibrationEvent:
    params: CalibrationParams
    trigger: str
    accepted: bool
    recorded_at: datetime
