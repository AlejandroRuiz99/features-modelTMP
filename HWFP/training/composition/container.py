"""Training composition root — pure factory functions, no global state."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from HWFP.core.application.promote_model import PromoteModelUseCase, PromotionGates
from HWFP.core.application.train_model import TrainModelUseCase
from HWFP.serving.adapters.filesystem_model_registry import FilesystemModelRegistry
from HWFP.serving.fakes import FakeModelRegistry
from HWFP.training.adapters.parquet_training_data_source import (
    ParquetTrainingDataSource,
)
from HWFP.training.adapters.pytorch_model_trainer import PyTorchModelTrainer
from HWFP.training.fakes import FakeModelTrainer, FakeTrainingDataSource
from HWFP.training.paths import default_training_data_path


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


def _resolve_training_data_path() -> Path:
    """Env `HWFP_TRAINING_DATA` override, else the package-default parquet."""
    override = os.environ.get("HWFP_TRAINING_DATA")
    return Path(override) if override else default_training_data_path()


def _resolve_checkpoints_dir() -> Path:
    """Env `HWFP_CHECKPOINTS_DIR` override, else HWFP.models' package default.

    Mirrors `HWFP.cli.bot_main._resolve_checkpoints_dir()` — both the
    serving and training composition roots resolve the same production
    checkpoints tree the same way.
    """
    from HWFP.models.paths import default_checkpoints_dir

    override = os.environ.get("HWFP_CHECKPOINTS_DIR")
    return Path(override) if override else default_checkpoints_dir()


def container_production_training() -> Tuple[TrainModelUseCase, PromoteModelUseCase]:
    """Wire train and promote use cases with real production adapters.

    - source: `ParquetTrainingDataSource` over `HWFP_TRAINING_DATA` (env
      override) or the package-default `HWFP/training/data/training.parquet`.
    - trainer: `PyTorchModelTrainer` — real `fit()`.
    - registry: `FilesystemModelRegistry` pointed at the production
      checkpoints dir. `register()` only ever writes new candidates under
      `candidates/{model_id}/` — it never overwrites the production
      checkpoints directory (non-negotiable per spec; `promote()` is the
      only sanctioned path there, and stays `NotImplementedError` — out of
      scope for this change, same as before).
    - clock: real UTC now.
    - git_sha_provider: env `GIT_SHA`, fallback `"unknown"` — no subprocess.
    """
    registry = FilesystemModelRegistry(checkpoints_dir=_resolve_checkpoints_dir())
    train_uc = TrainModelUseCase(
        source=ParquetTrainingDataSource(_resolve_training_data_path()),
        trainer=PyTorchModelTrainer(),
        registry=registry,
        clock=lambda: datetime.now(timezone.utc),
        git_sha_provider=lambda: os.environ.get("GIT_SHA", "unknown"),
    )
    promote_uc = PromoteModelUseCase(
        registry=registry,
        gates=PromotionGates(),
    )
    return train_uc, promote_uc
