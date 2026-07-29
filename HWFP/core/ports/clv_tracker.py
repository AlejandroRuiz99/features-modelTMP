"""CLVTracker port — record closing lines and compute CLV metrics."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CLVTracker(Protocol):
    """Track closing line value for placed bets."""

    def record_closing_line(
        self, bet_id: str, closing_line: float, closing_odds: float
    ) -> None: ...
    def compute_clv(self, bet_id: str) -> float | None: ...
    def get_avg_clv(self, last_n: int | None = None) -> float | None: ...
