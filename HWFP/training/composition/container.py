"""Training composition root — pure factory functions, no global state."""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

from HWFP.core.application.promote_model import PromoteModelUseCase, PromotionGates
from HWFP.core.application.train_model import TrainModelUseCase
from HWFP.serving.fakes import FakeModelRegistry
from HWFP.training.fakes import FakeModelTrainer, FakeTrainingDataSource


def container_fakes_training() -> Tuple[TrainModelUseCase, PromoteModelUseCase]:
    """Wire train and promote use cases with all fakes. Deterministic. Zero I/O."""
    registry = FakeModelRegistry()
    train_uc = TrainModelUseCase(
        source=FakeTrainingDataSource(),
        trainer=FakeModelTrainer(),
        registry=registry,
        clock=lambda: datetime(2026, 6, 16, 20, 0, 0),
        git_sha_provider=lambda: "fixture-sha-0000",
    )
    promote_uc = PromoteModelUseCase(
        registry=registry,
        gates=PromotionGates(),
    )
    return train_uc, promote_uc


def container_production_training() -> Tuple[TrainModelUseCase, PromoteModelUseCase]:
    """Raises until real training adapters are implemented."""
    raise NotImplementedError(
        "Production training adapters are stubs. Implement in: "
        "hwfp-csv-training-data-source-adapter, hwfp-pytorch-model-trainer-adapter."
    )
