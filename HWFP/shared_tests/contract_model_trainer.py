"""Contract tests for ModelTrainer port (REQ-8, REQ-9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.model_manifest import HoldoutMetrics
from HWFP.core.domain.training_example import TrainingExample

_EXAMPLES = [
    TrainingExample(
        match_id="M1",
        features=(0.1, 0.2, 0.3),
        actual_fouls=22,
        kickoff=datetime(2026, 6, 16, 20, 0),
    ),
]
_HYPERPARAMS: dict = {}


@pytest.fixture(params=["fake", "real"], ids=["fake", "real"])
def model_trainer(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.training.fakes.fake_model_trainer")
        return mod.FakeModelTrainer()
    mod = pytest.importorskip("HWFP.training.adapters.pytorch_model_trainer")
    return mod.PyTorchModelTrainer()


def test_fit_returns_two_element_tuple(model_trainer):
    result = model_trainer.fit(_EXAMPLES, _HYPERPARAMS)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_fit_blob_is_non_empty_bytes(model_trainer):
    blob, _ = model_trainer.fit(_EXAMPLES, _HYPERPARAMS)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_fit_metrics_is_holdout_metrics(model_trainer):
    _, metrics = model_trainer.fit(_EXAMPLES, _HYPERPARAMS)
    assert isinstance(metrics, HoldoutMetrics)


def test_fit_metrics_are_positive(model_trainer):
    _, metrics = model_trainer.fit(_EXAMPLES, _HYPERPARAMS)
    assert metrics.nll > 0.0
    assert metrics.brier > 0.0
    assert metrics.calibration_ece >= 0.0
