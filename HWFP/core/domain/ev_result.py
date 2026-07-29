from __future__ import annotations

from dataclasses import dataclass

from HWFP.core.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class EVResult:
    match_id: str
    market: str
    line: float
    side: str
    fair_prob: float
    book_prob: float
    ev: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.fair_prob <= 1.0):
            raise DomainValidationError(
                f"fair_prob must be in [0, 1], got {self.fair_prob}"
            )
        if not (0.0 < self.book_prob < 1.0):
            raise DomainValidationError(
                f"book_prob must be in (0, 1), got {self.book_prob}"
            )
