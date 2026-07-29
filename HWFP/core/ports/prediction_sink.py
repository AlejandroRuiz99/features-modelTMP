"""PredictionSink port — write a prediction to a storage backend."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.match_prediction import MatchPrediction


class PredictionSink(Protocol):
    """Write a MatchPrediction to a storage backend.

    Raises:
        SinkWriteError: If the write operation fails.
    """

    def write(self, prediction: MatchPrediction) -> None: ...
