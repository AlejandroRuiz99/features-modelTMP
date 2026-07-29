"""FakeTrainingDataSource — fixed deterministic training examples. Zero I/O."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from HWFP.core.domain.training_example import TrainingExample

_EXAMPLES: tuple[TrainingExample, ...] = (
    TrainingExample(
        match_id="M1",
        features=(0.1, 0.2, 0.3, 0.4),
        actual_fouls=22,
        kickoff=datetime(2026, 6, 14, 20, 0),
    ),
    TrainingExample(
        match_id="M2",
        features=(0.2, 0.3, 0.4, 0.5),
        actual_fouls=18,
        kickoff=datetime(2026, 6, 7, 18, 0),
    ),
    TrainingExample(
        match_id="M3",
        features=(0.3, 0.1, 0.2, 0.6),
        actual_fouls=25,
        kickoff=datetime(2026, 5, 31, 16, 0),
    ),
)

_DATASET_HASH = "hash-fixture-001"


class FakeTrainingDataSource:
    """Yields 3 fixed TrainingExample instances. Deterministic. Zero I/O.

    dataset_hash() returns "hash-fixture-001" consistently across calls.
    iter_examples() is repeatable: same instances, same order every time.
    """

    def iter_examples(self) -> Iterator[TrainingExample]:
        yield from _EXAMPLES

    def dataset_hash(self) -> str:
        return _DATASET_HASH
