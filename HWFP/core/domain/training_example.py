from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingExample:
    match_id: str
    features: tuple[float, ...]
    actual_fouls: int
    kickoff: datetime
    # Contingency (design D3, confirmed in Batch 5 RED): FoulPredictionEnsemble.fit()
    # needs far more than the 76 canonical numeric features per match — team/referee
    # identity, season, derby flag, etc. `metadata` carries the raw source row so
    # PyTorchModelTrainer can reconstruct the full feature dict without widening the
    # `features` tuple's meaning. Non-breaking: defaults to empty, existing callers
    # (FakeTrainingDataSource, contract tests) are unaffected.
    metadata: Mapping[str, Any] = field(default_factory=dict)
