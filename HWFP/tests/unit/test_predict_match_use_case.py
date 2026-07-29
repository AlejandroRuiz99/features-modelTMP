"""Unit tests for PredictMatchUseCase — B7a TDD RED gate."""

from __future__ import annotations

import math
from datetime import datetime

from HWFP.core.application.predict_match import PredictMatchInput, PredictMatchUseCase
from HWFP.serving.fakes.fake_ev_calculator import FakeEVCalculator
from HWFP.serving.fakes.fake_feature_builder import FakeFeatureBuilder
from HWFP.serving.fakes.fake_model_registry import FakeModelRegistry
from HWFP.serving.fakes.fake_odds_provider import FakeOddsProvider
from HWFP.serving.fakes.fake_overlay_engine import FakeOverlayEngine
from HWFP.serving.fakes.fake_prediction_sink import FakePredictionSink
from HWFP.serving.fakes.fake_referee_profiler import FakeRefereeProfiler
from HWFP.serving.fakes.fake_staking_calculator import FakeStakingCalculator
from HWFP.serving.fakes.fake_state_provider import FakeStateProvider

_CLOCK = lambda: datetime(2026, 6, 16, 20, 0, 0)  # noqa: E731

_INPUT = PredictMatchInput(
    match_id="M1",
    market="fouls_over_under",
    line=22.5,
    side="over",
    bankroll=1000.0,
)


def _build_use_case(sink: FakePredictionSink) -> PredictMatchUseCase:
    return PredictMatchUseCase(
        states=FakeStateProvider.with_fixture(),
        odds=FakeOddsProvider.with_fixture(),
        features=FakeFeatureBuilder(),
        model_registry=FakeModelRegistry.with_production_model(),
        referee=FakeRefereeProfiler.with_fixture(),
        overlay=FakeOverlayEngine(),
        ev_calc=FakeEVCalculator(),
        staking=FakeStakingCalculator(kelly_fraction=0.25),
        sink=sink,
        clock=_CLOCK,
    )


def test_predict_match_happy_path() -> None:
    out = _build_use_case(FakePredictionSink()).execute(_INPUT)

    assert abs(sum(out.prediction.pmf.pmf) - 1.0) <= 1e-6
    assert math.isfinite(out.ev.ev)
    assert out.stake.stake >= 0


def test_predict_match_sink_receives_prediction() -> None:
    sink = FakePredictionSink()
    _build_use_case(sink).execute(_INPUT)

    assert len(sink.writes) == 1
    assert sink.writes[0].match_id == "M1"
