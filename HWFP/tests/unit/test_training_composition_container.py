"""Unit tests for the training composition root (Batch 5, task 5.9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from HWFP.core.application.train_model import TrainInput
from HWFP.core.domain.model_status import ModelStatus
from HWFP.training.composition.container import container_production_training

_REAL_PARQUET = Path("prediction_models/data/training.parquet")
_MOVED_PARQUET = Path("HWFP/training/data/training.parquet")


def _source_parquet() -> Path:
    return _MOVED_PARQUET if _MOVED_PARQUET.exists() else _REAL_PARQUET


def _write_fixture_slice(path: Path) -> None:
    df = pd.read_parquet(_source_parquet())
    frames = [
        df[df["season"] == season].head(n)
        for season, n in {"2023-24": 8, "2024-25": 6, "2025-26": 6}.items()
    ]
    pd.concat(frames, ignore_index=True).to_parquet(path)


def test_container_production_training_registers_candidate_never_touches_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runtime harness: container_production_training() wired end to end
    registers a real trained candidate under candidates/, and never
    creates/writes the production checkpoints directory itself."""
    fixture_path = tmp_path / "fixture_training.parquet"
    _write_fixture_slice(fixture_path)

    checkpoints_dir = tmp_path / "checkpoints" / "ensemble"
    monkeypatch.setenv("HWFP_TRAINING_DATA", str(fixture_path))
    monkeypatch.setenv("HWFP_CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("GIT_SHA", "testsha123")

    train_uc, _ = container_production_training()
    manifest = train_uc.execute(
        TrainInput(
            hyperparams={
                "model_config": {"anfis": {"epochs": 2}},
                "no_grid_search": True,
                "seed": 3,
            }
        )
    )

    assert manifest.git_sha == "testsha123"
    assert manifest.status == ModelStatus.CANDIDATE
    assert manifest.dataset_rows == 20

    candidate_dir = checkpoints_dir.parent / "candidates" / manifest.model_id.value
    assert candidate_dir.exists()
    assert (candidate_dir / "config.json").exists()
    assert (candidate_dir / "manifest.json").exists()
    assert not checkpoints_dir.exists()


def test_resolve_training_data_path_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from HWFP.training.composition.container import _resolve_training_data_path

    fake_path = tmp_path / "custom.parquet"
    monkeypatch.setenv("HWFP_TRAINING_DATA", str(fake_path))
    assert _resolve_training_data_path() == fake_path


def test_resolve_training_data_path_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from HWFP.training.composition.container import _resolve_training_data_path
    from HWFP.training.paths import default_training_data_path

    monkeypatch.delenv("HWFP_TRAINING_DATA", raising=False)
    assert _resolve_training_data_path() == default_training_data_path()
