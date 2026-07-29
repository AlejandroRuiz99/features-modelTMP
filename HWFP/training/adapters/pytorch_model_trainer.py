"""Adapter stub — PyTorchModelTrainer (port: ModelTrainer)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from HWFP.core.domain.model_manifest import HoldoutMetrics
from HWFP.core.domain.training_example import TrainingExample


class PyTorchModelTrainer:
    """Stub for ModelTrainer. Raises NotImplementedError on all methods."""

    def fit(
        self,
        examples: List[TrainingExample],
        hyperparams: Dict,
    ) -> Tuple[bytes, HoldoutMetrics]:
        raise NotImplementedError(
            "HWFP adapter stub — port=ModelTrainer; future change hwfp-pytorch-model-trainer-adapter"
        )
