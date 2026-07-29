from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from HWFP.core.domain.exceptions import DomainValidationError

_VALID_SIDES: frozenset[str] = frozenset({"over", "under"})


@dataclass(frozen=True)
class Odds:
    match_id: str
    market: str
    line: float
    side: str
    decimal: float
    bookmaker: str
    fetched_at: datetime

    def __post_init__(self) -> None:
        if self.decimal <= 1.0:
            raise DomainValidationError(
                f"Odds.decimal must be > 1.0, got {self.decimal}"
            )
        if self.side not in _VALID_SIDES:
            raise DomainValidationError(
                f"Odds.side must be 'over' or 'under', got '{self.side}'"
            )
