"""Contract tests for TrainingDataSource port (REQ-8, REQ-9)."""

from __future__ import annotations

import pytest

from HWFP.core.domain.training_example import TrainingExample


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def training_data_source(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.training.fakes.fake_training_data_source")
        return mod.FakeTrainingDataSource()
    pytest.importorskip("HWFP.training.adapters.csv_training_data_source")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_iter_examples_yields_training_examples(training_data_source):
    examples = list(training_data_source.iter_examples())
    assert len(examples) > 0
    assert all(isinstance(e, TrainingExample) for e in examples)


def test_dataset_hash_returns_non_empty_string(training_data_source):
    h = training_data_source.dataset_hash()
    assert isinstance(h, str)
    assert len(h) > 0


def test_dataset_hash_is_stable(training_data_source):
    assert training_data_source.dataset_hash() == training_data_source.dataset_hash()


def test_iter_examples_is_repeatable(training_data_source):
    first = list(training_data_source.iter_examples())
    second = list(training_data_source.iter_examples())
    assert first == second


def test_training_data_source_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.training.adapters.csv_training_data_source")
    adapter = mod.CsvTrainingDataSource()
    with pytest.raises(NotImplementedError):
        adapter.dataset_hash()
