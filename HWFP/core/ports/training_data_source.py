"""TrainingDataSource port — iterate training examples."""

from __future__ import annotations

from typing import Iterator, Protocol

from HWFP.core.domain.training_example import TrainingExample


class TrainingDataSource(Protocol):
    """Lazy iteration over training examples and stable dataset hash.

    Raises:
        TrainingDataError: If data access or iteration fails.
    """

    def iter_examples(self) -> Iterator[TrainingExample]: ...

    def dataset_hash(self) -> str: ...
