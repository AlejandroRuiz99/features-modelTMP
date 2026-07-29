"""ModelTrainer port — fit a foul prediction model (12th port, training-only)."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from HWFP.core.domain.model_manifest import HoldoutMetrics
from HWFP.core.domain.training_example import TrainingExample


class ModelTrainer(Protocol):
    """Fit a foul prediction model from labelled training examples.

    Returns:
        Tuple of (model_blob, holdout_metrics) where model_blob is an opaque
        bytes payload understood by the serving adapter.
    """

    def fit(
        self,
        examples: List[TrainingExample],
        hyperparams: Dict[str, Any],
    ) -> Tuple[bytes, HoldoutMetrics]: ...
