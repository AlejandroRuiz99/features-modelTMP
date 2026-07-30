"""Contract tests for TrainingDataSource port (REQ-8, REQ-9)."""

from __future__ import annotations

import pytest

from HWFP.core.domain.training_example import TrainingExample

_FIXTURE_ROWS = [
    {
        "date": "2023-08-11",
        "season": "2023-24",
        "home_team": "Barcelona",
        "away_team": "Getafe",
        "referee": "Munuera Montero",
        "is_derby": False,
        "matchday": 1,
        "fouls_total": 24.0,
    },
    {
        "date": "2023-08-12",
        "season": "2023-24",
        "home_team": "Sevilla",
        "away_team": "Valencia",
        "referee": "Hernandez Hernandez",
        "is_derby": False,
        "matchday": 1,
        "fouls_total": 28.0,
    },
]


@pytest.fixture(params=["fake", "real"], ids=["fake", "real"])
def training_data_source(request, tmp_path):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.training.fakes.fake_training_data_source")
        return mod.FakeTrainingDataSource()
    mod = pytest.importorskip(
        "HWFP.training.adapters.parquet_training_data_source"
    )
    import pandas as pd

    path = tmp_path / "contract_training_fixture.parquet"
    pd.DataFrame(_FIXTURE_ROWS).to_parquet(path)
    return mod.ParquetTrainingDataSource(path)


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
