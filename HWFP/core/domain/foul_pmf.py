from __future__ import annotations

from dataclasses import dataclass

from HWFP.core.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class FoulPMF:
    pmf: tuple[float, ...]
    bin_edges: tuple[int, ...]

    def __post_init__(self) -> None:
        if abs(sum(self.pmf) - 1.0) > 1e-6:
            raise DomainValidationError(
                f"pmf must sum to 1.0 ± 1e-6, got {sum(self.pmf)}"
            )
        if not all(0.0 <= p <= 1.0 for p in self.pmf):
            raise DomainValidationError("all pmf values must be in [0.0, 1.0]")
        if len(self.bin_edges) != len(self.pmf) + 1:
            raise DomainValidationError(
                f"len(bin_edges) must equal len(pmf)+1: "
                f"got {len(self.bin_edges)} vs {len(self.pmf) + 1}"
            )
