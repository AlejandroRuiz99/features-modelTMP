"""T8.1 — Golden e2e: full predict pipeline wired with all fakes."""

from __future__ import annotations

import pytest

from HWFP.core.application.predict_match import PredictMatchInput
from HWFP.serving.composition.container import container_fakes
from HWFP.serving.fakes import FakePredictionSink


def test_predict_all_fake_deterministic_output() -> None:
    use_case = container_fakes()
    inp = PredictMatchInput(
        match_id="M1",
        market="fouls_over_under",
        line=22.5,
        side="over",
        bankroll=1000.0,
    )
    out = use_case.execute(inp)

    # PMF: exact fixed values from FakeFoulModel
    assert out.prediction.pmf.pmf == (0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05)

    # fair_prob: bins where bin_edges[i+1] > 22.5 → bins 3,4,5,6 → 0.30+0.20+0.10+0.05 = 0.65
    assert out.ev.fair_prob == pytest.approx(0.65, abs=1e-9)

    # ev = 1.95 × 0.65 − 1.0 = 0.2675
    assert out.ev.ev == pytest.approx(0.2675, abs=1e-9)

    # stake = 0.25 × 1000.0 = 250.0
    assert out.stake.stake == pytest.approx(250.0, abs=1e-9)

    # Sink received exactly one entry
    assert isinstance(use_case._sink, FakePredictionSink)
    assert len(use_case._sink.writes) == 1
    assert use_case._sink.writes[0].match_id == "M1"
