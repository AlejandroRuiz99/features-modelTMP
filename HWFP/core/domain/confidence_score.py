from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ConfidenceScore:
    pmf_entropy: float
    referee_sample_size: int
    feature_fallback_count: int
    kelly_multiplier: float
    level: ConfidenceLevel
