"""Adapter — ParquetTrainingDataSource (port: TrainingDataSource)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import pandas as pd

from HWFP.core.domain.feature_keys import CANONICAL_FEATURE_KEYS
from HWFP.core.domain.training_example import TrainingExample


class ParquetTrainingDataSource:
    """Loads training examples from a Parquet file. Read-only, deterministic.

    ``iter_examples()`` extracts the 76 ``CANONICAL_FEATURE_KEYS`` as the
    ``features`` tuple (defaulting missing keys to 0.0, matching
    ``PyTorchFeatureBuilder.build()``'s convention) and carries the full raw
    row in ``metadata`` — ``FoulPredictionEnsemble.fit()`` needs team/referee
    identity, season, and the ``fouls_total`` label, which the 76-feature
    tuple alone cannot carry (design D3 contingency, confirmed in RED).

    ``dataset_hash()`` is a sha256 of the file's raw bytes: stable across
    repeated calls, changes whenever the file's content changes.
    """

    def __init__(self, parquet_path: str | Path) -> None:
        self._path = Path(parquet_path)

    def iter_examples(self) -> Iterator[TrainingExample]:
        df = pd.read_parquet(self._path)
        for _, row in df.iterrows():
            record = row.to_dict()
            features = tuple(
                float(record.get(key, 0.0)) for key in CANONICAL_FEATURE_KEYS
            )
            kickoff = pd.Timestamp(record["date"]).to_pydatetime()
            match_id = (
                f"{record.get('home_team', '?')}-{record.get('away_team', '?')}-"
                f"{record.get('date', '?')}"
            )
            yield TrainingExample(
                match_id=str(match_id),
                features=features,
                actual_fouls=int(record["fouls_total"]),
                kickoff=kickoff,
                metadata=record,
            )

    def dataset_hash(self) -> str:
        return hashlib.sha256(self._path.read_bytes()).hexdigest()
