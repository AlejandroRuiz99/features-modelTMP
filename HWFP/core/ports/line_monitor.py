"""LineMonitor port — fetch live odds and detect line movements."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from HWFP.core.domain.line_movement import LineMovement
from HWFP.core.domain.odds import Odds


@runtime_checkable
class LineMonitor(Protocol):
    """Monitor live odds and detect significant line movements."""

    def get_current_odds(self, match_id: str) -> list[Odds]: ...
    def detect_movements(self, match_id: str, since: datetime) -> list[LineMovement]: ...
    def get_significant_movements(self, since: datetime) -> list[LineMovement]: ...
