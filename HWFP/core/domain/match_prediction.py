from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.model_id import ModelId


@dataclass(frozen=True)
class MatchPrediction:
    match_id: str
    pmf: FoulPMF
    model_id: ModelId
    generated_at: datetime
