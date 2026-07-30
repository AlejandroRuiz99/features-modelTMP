"""Training adapters — production TrainingDataSource and ModelTrainer implementations."""

from __future__ import annotations

from HWFP.training.adapters.parquet_training_data_source import (
    ParquetTrainingDataSource,
)
from HWFP.training.adapters.pytorch_model_trainer import PyTorchModelTrainer

__all__ = [
    "ParquetTrainingDataSource",
    "PyTorchModelTrainer",
]
