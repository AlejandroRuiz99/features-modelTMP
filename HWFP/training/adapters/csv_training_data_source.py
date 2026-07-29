"""Adapter stub — CsvTrainingDataSource (port: TrainingDataSource)."""

from __future__ import annotations

from typing import Iterator

from HWFP.core.domain.training_example import TrainingExample


class CsvTrainingDataSource:
    """Stub for TrainingDataSource. Raises NotImplementedError on all methods."""

    def iter_examples(self) -> Iterator[TrainingExample]:
        raise NotImplementedError(
            "HWFP adapter stub — port=TrainingDataSource; future change hwfp-csv-training-data-source-adapter"
        )

    def dataset_hash(self) -> str:
        raise NotImplementedError(
            "HWFP adapter stub — port=TrainingDataSource; future change hwfp-csv-training-data-source-adapter"
        )
