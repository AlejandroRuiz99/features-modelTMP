"""Training fakes — deterministic in-memory implementations for all training ports."""

from __future__ import annotations

from HWFP.training.fakes.fake_model_trainer import FakeModelTrainer
from HWFP.training.fakes.fake_training_data_source import FakeTrainingDataSource

__all__ = [
    "FakeModelTrainer",
    "FakeTrainingDataSource",
]
