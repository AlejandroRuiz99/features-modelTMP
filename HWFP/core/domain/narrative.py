from __future__ import annotations

from dataclasses import dataclass

from HWFP.core.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class Narrative:
    match_id: str
    text: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.text:
            raise DomainValidationError("Narrative.text cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise DomainValidationError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
