"""Contract tests for PredictionSink port (REQ-8, REQ-9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.match_prediction import MatchPrediction
from HWFP.core.domain.model_id import ModelId

_PREDICTION = MatchPrediction(
    match_id="M1",
    pmf=FoulPMF(
        pmf=(0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05),
        bin_edges=(0, 15, 20, 22, 24, 26, 30, 40),
    ),
    model_id=ModelId("fake-model-001"),
    generated_at=datetime(2026, 6, 16, 20, 0, 0),
)


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def prediction_sink(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_prediction_sink")
        return mod.FakePredictionSink()
    pytest.importorskip("HWFP.serving.adapters.supabase_prediction_sink")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_write_accepts_match_prediction(prediction_sink):
    prediction_sink.write(_PREDICTION)


def test_write_twice_no_exception(prediction_sink):
    prediction_sink.write(_PREDICTION)
    prediction_sink.write(_PREDICTION)


def test_prediction_sink_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.supabase_prediction_sink")
    adapter = mod.SupabasePredictionSink()
    with pytest.raises(NotImplementedError):
        adapter.write(_PREDICTION)
