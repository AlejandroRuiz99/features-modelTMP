from __future__ import annotations

import re
from dataclasses import dataclass

from HWFP.core.domain.exceptions import DomainValidationError

_PATTERN = re.compile(r"^[a-z0-9_-]{3,64}$")


@dataclass(frozen=True)
class ModelId:
    value: str

    def __post_init__(self) -> None:
        if not _PATTERN.match(self.value):
            raise DomainValidationError(
                f"ModelId '{self.value}' must match ^[a-z0-9_-]{{3,64}}$"
            )
