"""Unit tests for TrainModelUseCase — B7b TDD RED gate (T7b.3)."""

from __future__ import annotations

from datetime import datetime

from HWFP.core.application.train_model import TrainInput, TrainModelUseCase
from HWFP.core.domain.model_status import ModelStatus
from HWFP.serving.fakes.fake_model_registry import FakeModelRegistry
from HWFP.training.fakes.fake_model_trainer import FakeModelTrainer
from HWFP.training.fakes.fake_training_data_source import FakeTrainingDataSource


def _build_uc(registry: FakeModelRegistry | None = None) -> TrainModelUseCase:
    return TrainModelUseCase(
        source=FakeTrainingDataSource(),
        trainer=FakeModelTrainer(),
        registry=registry if registry is not None else FakeModelRegistry(),
        clock=lambda: datetime(2026, 6, 16, 20, 0, 0),
        git_sha_provider=lambda: "abc123de",
    )


def test_train_model_returns_candidate_manifest() -> None:
    """T7b.3: execute() returns a manifest with status=CANDIDATE."""
    manifest = _build_uc().execute(TrainInput())
    assert manifest.status == ModelStatus.CANDIDATE


def test_train_model_manifest_appears_in_registry() -> None:
    """T7b.3: returned manifest appears in registry list_candidates."""
    registry = FakeModelRegistry()
    manifest = _build_uc(registry=registry).execute(TrainInput())

    candidates = registry.list_manifests(status=ModelStatus.CANDIDATE)
    assert manifest in candidates


def test_train_model_dataset_hash_propagated() -> None:
    """T7b.3: dataset_hash is taken from the source, not hardcoded."""
    manifest = _build_uc().execute(TrainInput())
    assert manifest.dataset_hash == "hash-fixture-001"


def test_train_model_dataset_rows_matches_source() -> None:
    """FakeTrainingDataSource has 3 examples."""
    manifest = _build_uc().execute(TrainInput())
    assert manifest.dataset_rows == 3


def test_train_model_metrics_from_trainer() -> None:
    """HoldoutMetrics come from FakeModelTrainer.fit()."""
    manifest = _build_uc().execute(TrainInput())
    assert manifest.metrics_holdout.nll == 0.5
    assert manifest.metrics_holdout.brier == 0.18
