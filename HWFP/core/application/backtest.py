"""BacktestUseCase — walk-forward evaluation over historical examples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

from HWFP.core.application.predict_match import PredictMatchInput, PredictMatchUseCase
from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.ports.training_data_source import TrainingDataSource


@dataclass
class BacktestInput:
    start_date: datetime
    end_date: datetime
    market: str
    line: float
    side: str
    bankroll: float


@dataclass
class BacktestOutput:
    ev_results: Tuple[EVResult, ...]
    hit_rate: float
    avg_nll: float


def _nll(pmf: FoulPMF, actual_fouls: int) -> float:
    """Negative log-likelihood of actual_fouls landing in its PMF bin."""
    for i in range(len(pmf.pmf) - 1):
        if pmf.bin_edges[i] <= actual_fouls < pmf.bin_edges[i + 1]:
            return -math.log(max(pmf.pmf[i], 1e-12))
    return -math.log(max(pmf.pmf[-1], 1e-12))


class BacktestUseCase:
    def __init__(
        self,
        predict: PredictMatchUseCase,
        source: TrainingDataSource,
    ) -> None:
        self._predict = predict
        self._source = source

    def execute(self, inp: BacktestInput) -> BacktestOutput:
        ev_results = []
        nll_values = []
        hits = 0

        for example in self._source.iter_examples():
            if not (inp.start_date <= example.kickoff <= inp.end_date):
                continue

            out = self._predict.execute(
                PredictMatchInput(
                    match_id=example.match_id,
                    market=inp.market,
                    line=inp.line,
                    side=inp.side,
                    bankroll=inp.bankroll,
                )
            )
            ev_results.append(out.ev)
            nll_values.append(_nll(out.prediction.pmf, example.actual_fouls))

            if inp.side == "over" and example.actual_fouls > inp.line:
                hits += 1
            elif inp.side == "under" and example.actual_fouls < inp.line:
                hits += 1

        total = len(ev_results)
        return BacktestOutput(
            ev_results=tuple(ev_results),
            hit_rate=hits / total if total > 0 else 0.0,
            avg_nll=sum(nll_values) / len(nll_values) if nll_values else 0.0,
        )
