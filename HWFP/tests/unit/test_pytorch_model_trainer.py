"""Unit tests for PyTorchModelTrainer (port: ModelTrainer, Batch 5)."""

from __future__ import annotations

import io
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
import torch

from HWFP.core.domain.feature_keys import CANONICAL_FEATURE_KEYS
from HWFP.core.domain.model_manifest import HoldoutMetrics
from HWFP.core.domain.training_example import TrainingExample
from HWFP.training.adapters.parquet_training_data_source import (
    ParquetTrainingDataSource,
)
from HWFP.training.adapters.pytorch_model_trainer import (
    PyTorchModelTrainer,
    _match_dict,
    _team_averages,
)

_REAL_PARQUET = Path("HWFP/training/data/training.parquet")
_FAST_MODEL_CONFIG = {"anfis": {"epochs": 2}}


def _examples_from_dataframe(df: pd.DataFrame) -> Iterator[TrainingExample]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slice.parquet"
        df.to_parquet(path)
        yield from ParquetTrainingDataSource(path).iter_examples()


def _load_examples(seasons_and_counts: dict) -> list[TrainingExample]:
    df = pd.read_parquet(_REAL_PARQUET)
    frames = [
        df[df["season"] == season].head(n) for season, n in seasons_and_counts.items()
    ]
    slice_df = pd.concat(frames, ignore_index=True)
    return list(_examples_from_dataframe(slice_df))


@pytest.fixture(scope="module")
def small_examples() -> list[TrainingExample]:
    """~26 real matches across 3 seasons — enough for the season-based
    temporal split (>=3 distinct seasons) without the size of a full
    production run (kept small + low ANFIS epochs for CI speed)."""
    return _load_examples({"2023-24": 10, "2024-25": 8, "2025-26": 8})


# ---------------------------------------------------------------------------
# Confirms design D3's contingency (metadata beyond the 76 features)
# ---------------------------------------------------------------------------


def test_team_averages_require_metadata_team_identity() -> None:
    """RED confirmation (design D3 contingency): the 76 canonical features
    carry no team identity. FoulPredictionEnsemble.fit()'s per-team Naive
    Bayes priors are built from `home_team`/`away_team`, only available via
    TrainingExample.metadata — without it, _team_averages() degrades to
    empty (zero team-specific signal at all), proving metadata is not
    optional decoration but load-bearing for real training quality."""
    bare = TrainingExample(
        match_id="bare",
        features=tuple(0.0 for _ in CANONICAL_FEATURE_KEYS),
        actual_fouls=20,
        kickoff=datetime(2023, 8, 1),
    )
    with_identity = TrainingExample(
        match_id="real",
        features=tuple(0.0 for _ in CANONICAL_FEATURE_KEYS),
        actual_fouls=20,
        kickoff=datetime(2023, 8, 1),
        metadata={
            "home_team": "Barcelona",
            "away_team": "Getafe",
            "home_fouls_committed_avg": 11.0,
            "away_fouls_committed_avg": 13.5,
        },
    )

    committed_bare, _, _ = _team_averages([_match_dict(bare)])
    committed_real, _, _ = _team_averages([_match_dict(with_identity)])

    assert committed_bare == {}
    assert committed_real == {"Barcelona": 11.0, "Getafe": 13.5}


# ---------------------------------------------------------------------------
# Scenario "End-to-end training"
# ---------------------------------------------------------------------------


def test_fit_returns_bytes_and_holdout_metrics(small_examples) -> None:
    trainer = PyTorchModelTrainer()
    blob, metrics = trainer.fit(
        small_examples,
        {"model_config": _FAST_MODEL_CONFIG, "no_grid_search": True, "seed": 7},
    )
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    assert isinstance(metrics, HoldoutMetrics)
    assert metrics.nll > 0.0
    assert metrics.brier >= 0.0
    assert metrics.calibration_ece >= 0.0


def test_fit_checkpoint_round_trips_and_predicts(small_examples, tmp_path) -> None:
    """Runtime harness: the produced blob is a real, loadable checkpoint that
    predicts successfully — not a mocked/short-circuited artifact. Also
    proves the checkpoint self-describes via config.json (Batch 4's
    checkpoint-authoritative load contract)."""
    from HWFP.models.ensemble import FoulPredictionEnsemble

    trainer = PyTorchModelTrainer()
    blob, _ = trainer.fit(
        small_examples,
        {"model_config": _FAST_MODEL_CONFIG, "no_grid_search": True, "seed": 7},
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(checkpoint_dir)

    assert (checkpoint_dir / "config.json").exists()

    loaded = FoulPredictionEnsemble()
    loaded.load(checkpoint_dir)

    sample_match = _match_dict(small_examples[0])
    prediction = loaded.predict(sample_match)
    assert prediction.pmf_total.probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert prediction.expected_fouls > 0.0


def test_fit_persists_grid_search_min_weight_in_config(small_examples) -> None:
    """min_weight has no persistence path other than config.json (it is a
    plain DynamicEnsembleWeighter attribute, not part of gating.pt's
    torch state_dict) — a grid-search-selected value must survive reload."""
    trainer = PyTorchModelTrainer()
    blob, _ = trainer.fit(
        small_examples,
        {
            "model_config": _FAST_MODEL_CONFIG,
            "seed": 7,
            "grid_search_min_matches": 1,
            "grid": {
                "prior_mix": [0.42],
                "min_weight": [0.07],
                "variance_scale": [0.5],
            },
        },
    )
    import json

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        config = json.loads(zf.read("config.json"))
    assert config["gating_network"]["min_weight"] == pytest.approx(0.07)
    assert config["gating_network"]["prior_mix"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Scenario "Equivalence tolerance"
# ---------------------------------------------------------------------------


def test_equivalence_against_scripts_train_within_tolerance() -> None:
    """fit()'s orchestration matches doing the same steps by hand with
    scripts/train.py's own ported functions (temporal_split, _team_averages)
    on identical data + identical seed. Tolerance pinned by the tasks
    artifact: |Δnll| <= 0.01. brier/calibration_ece have no scripts/train.py
    reference to compare against — it never computed them (see
    _evaluate_holdout's docstring for why HoldoutMetrics needed a new,
    documented formula) — so this test validates nll parity, the one
    metric both sides can compute identically.

    scripts.train is imported here (test-only) to reuse its exact reference
    functions; production code (pytorch_model_trainer.py) never imports it —
    that module mutates sys.path at import time, which is banned from
    anything reachable via HWFP/'s own import graph (REQ-14).
    """
    import scripts.train as legacy

    examples = _load_examples({"2023-24": 10, "2024-25": 8, "2025-26": 8})
    seed = 11
    fast_config = {"anfis": {"epochs": 2}}

    df = pd.DataFrame([dict(e.metadata) for e in examples])
    train_seasons = [s for s in sorted(df["season"].unique()) if s < "2024-25"]
    train_df, tune_df, test_df = legacy.temporal_split(
        df, train_seasons, "2024-25", "2025-26"
    )
    train_matches = train_df.to_dict("records")
    tune_matches = tune_df.to_dict("records")
    test_matches = test_df.to_dict("records")

    team_committed, team_suffered, team_rank = legacy._team_averages(train_matches)

    torch.manual_seed(seed)
    np.random.seed(seed)
    from HWFP.models.ensemble import FoulPredictionEnsemble

    ref_ensemble = FoulPredictionEnsemble(fast_config)
    ref_ensemble.fit(
        train_matches,
        team_avg_committed=team_committed,
        team_avg_suffered=team_suffered,
        team_avg_rank=team_rank,
        fit_team_models=True,
        gating_matches=tune_matches,
    )
    ref_ensemble.calibrate_fit(tune_matches)
    ref_eval = legacy._evaluate(ref_ensemble, test_matches)

    trainer = PyTorchModelTrainer()
    _, metrics = trainer.fit(
        examples,
        {
            "model_config": fast_config,
            "seed": seed,
            "no_grid_search": True,
            "tune_season": "2024-25",
            "test_season": "2025-26",
        },
    )

    assert abs(metrics.nll - ref_eval["nll"]) <= 0.01
