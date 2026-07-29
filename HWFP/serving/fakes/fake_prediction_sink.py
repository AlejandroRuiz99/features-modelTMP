"""FakePredictionSink — in-memory accumulator. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.match_prediction import MatchPrediction


class FakePredictionSink:
    """Accumulates written predictions in self.writes. No persistence.

    E2E assertion: len(sink.writes) == 1 after one predict call.
    """

    def __init__(self) -> None:
        self.writes: list[MatchPrediction] = []

    def write(self, prediction: MatchPrediction) -> None:
        self.writes.append(prediction)
