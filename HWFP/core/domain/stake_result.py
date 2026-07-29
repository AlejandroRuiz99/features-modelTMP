from __future__ import annotations

from dataclasses import dataclass

from HWFP.core.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class StakeResult:
    match_id: str
    market: str
    stake: float
    kelly_fraction: float
    bankroll_used: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.stake <= self.bankroll_used):
            raise DomainValidationError(
                f"stake {self.stake} must be in [0, bankroll_used={self.bankroll_used}]"
            )
        if not (0.0 <= self.kelly_fraction <= 1.0):
            raise DomainValidationError(
                f"kelly_fraction {self.kelly_fraction} must be in [0, 1]"
            )
