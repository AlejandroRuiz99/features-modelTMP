"""Unit tests for ParquetTrainingDataSource (port: TrainingDataSource, Batch 5)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from HWFP.core.domain.feature_keys import CANONICAL_FEATURE_KEYS
from HWFP.core.domain.training_example import TrainingExample
from HWFP.training.adapters.parquet_training_data_source import (
    ParquetTrainingDataSource,
)

_ROWS = [
    {
        "date": "2023-08-11",
        "season": "2023-24",
        "home_team": "Barcelona",
        "away_team": "Getafe",
        "referee": "Munuera Montero",
        "is_derby": False,
        "matchday": 1,
        "fouls_total": 24.0,
        "home_fouls_committed_avg": 11.0,
        "away_fouls_committed_avg": 13.5,
        "home_rank_curr": 1,
        "away_rank_curr": 14,
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
        "home_fouls_committed_avg": 12.2,
        "away_fouls_committed_avg": 14.1,
        "home_rank_curr": 8,
        "away_rank_curr": 9,
    },
]


@pytest.fixture
def fixture_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "training_fixture.parquet"
    pd.DataFrame(_ROWS).to_parquet(path)
    return path


def test_iter_examples_yields_training_examples_with_kickoff_and_76_features(
    fixture_parquet: Path,
) -> None:
    source = ParquetTrainingDataSource(fixture_parquet)
    examples = list(source.iter_examples())

    assert len(examples) == 2
    for example in examples:
        assert isinstance(example, TrainingExample)
        assert isinstance(example.kickoff, datetime)
        assert len(example.features) == len(CANONICAL_FEATURE_KEYS)
        assert all(isinstance(v, float) for v in example.features)


def test_iter_examples_features_match_canonical_key_values(
    fixture_parquet: Path,
) -> None:
    """Scenario 'Load examples': features align positionally with CANONICAL_FEATURE_KEYS."""
    source = ParquetTrainingDataSource(fixture_parquet)
    examples = list(source.iter_examples())

    first = examples[0]
    idx = CANONICAL_FEATURE_KEYS.index("home_fouls_committed_avg")
    assert first.features[idx] == pytest.approx(11.0)
    idx_away = CANONICAL_FEATURE_KEYS.index("away_fouls_committed_avg")
    assert first.features[idx_away] == pytest.approx(13.5)
    # Keys absent from the fixture row default to 0.0 (matches
    # PyTorchFeatureBuilder.build()'s convention).
    idx_missing = CANONICAL_FEATURE_KEYS.index("h2h_faltas_media")
    assert first.features[idx_missing] == pytest.approx(0.0)


def test_iter_examples_carries_raw_row_in_metadata(fixture_parquet: Path) -> None:
    """Scenario confirmed in RED: ensemble.fit() needs team/referee identity
    beyond the 76 canonical features — metadata carries the raw source row."""
    source = ParquetTrainingDataSource(fixture_parquet)
    examples = list(source.iter_examples())

    first = examples[0]
    assert first.metadata["home_team"] == "Barcelona"
    assert first.metadata["away_team"] == "Getafe"
    assert first.metadata["referee"] == "Munuera Montero"
    assert first.metadata["season"] == "2023-24"
    assert first.actual_fouls == 24


def test_dataset_hash_returns_non_empty_string(fixture_parquet: Path) -> None:
    source = ParquetTrainingDataSource(fixture_parquet)
    h = source.dataset_hash()
    assert isinstance(h, str)
    assert len(h) > 0


def test_dataset_hash_is_stable(fixture_parquet: Path) -> None:
    source = ParquetTrainingDataSource(fixture_parquet)
    assert source.dataset_hash() == source.dataset_hash()


def test_dataset_hash_changes_on_content_change(
    fixture_parquet: Path, tmp_path: Path
) -> None:
    """Scenario 'Hash stability': a modified file yields a different hash."""
    source = ParquetTrainingDataSource(fixture_parquet)
    original_hash = source.dataset_hash()

    modified_rows = _ROWS + [{**_ROWS[0], "fouls_total": 99.0}]
    modified_path = tmp_path / "training_fixture_modified.parquet"
    pd.DataFrame(modified_rows).to_parquet(modified_path)

    modified_source = ParquetTrainingDataSource(modified_path)
    assert modified_source.dataset_hash() != original_hash


def test_iter_examples_is_repeatable(fixture_parquet: Path) -> None:
    source = ParquetTrainingDataSource(fixture_parquet)
    first = list(source.iter_examples())
    second = list(source.iter_examples())
    assert first == second
