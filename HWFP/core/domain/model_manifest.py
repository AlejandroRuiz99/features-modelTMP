from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_status import ModelStatus


@dataclass(frozen=True)
class HoldoutMetrics:
    nll: float
    brier: float
    calibration_ece: float


@dataclass(frozen=True)
class ModelManifest:
    model_id: ModelId
    trained_at: datetime
    git_sha: str
    dataset_hash: str
    dataset_rows: int
    metrics_holdout: HoldoutMetrics
    gates_passed: tuple[str, ...]
    status: ModelStatus
