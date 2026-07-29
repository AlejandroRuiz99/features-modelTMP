"""Unit tests for BacktestUseCase — B7a TDD RED gate."""

from __future__ import annotations

from datetime import datetime

from HWFP.core.application.backtest import BacktestInput, BacktestUseCase
from HWFP.core.application.predict_match import PredictMatchUseCase
from HWFP.serving.fakes.fake_ev_calculator import FakeEVCalculator
from HWFP.serving.fakes.fake_feature_builder import FakeFeatureBuilder
from HWFP.serving.fakes.fake_model_registry import FakeModelRegistry
from HWFP.serving.fakes.fake_odds_provider import FakeOddsProvider
from HWFP.serving.fakes.fake_overlay_engine import FakeOverlayEngine
from HWFP.serving.fakes.fake_prediction_sink import FakePredictionSink
from HWFP.serving.fakes.fake_referee_profiler import FakeRefereeProfiler
from HWFP.serving.fakes.fake_staking_calculator import FakeStakingCalculator
from HWFP.serving.fakes.fake_state_provider import FakeStateProvider
from HWFP.training.fakes.fake_training_data_source import FakeTrainingDataSource


def _build_predict_uc() -> PredictMatchUseCase:
    return PredictMatchUseCase(
        states=FakeStateProvider.with_fixture(),
        odds=FakeOddsProvider.with_fixture(),
        features=FakeFeatureBuilder(),
        model_registry=FakeModelRegistry.with_production_model(),
        referee=FakeRefereeProfiler.with_fixture(),
        overlay=FakeOverlayEngine(),
        ev_calc=FakeEVCalculator(),
        staking=FakeStakingCalculator(kelly_fraction=0.25),
        sink=FakePredictionSink(),
        clock=lambda: datetime(2026, 6, 16, 20, 0, 0),
    )


# Window that contains only M1 (kickoff 2026-06-14T20:00)
_WINDOW_M1 = BacktestInput(
    start_date=datetime(2026, 6, 14),
    end_date=datetime(2026, 6, 15),
    market="fouls_over_under",
    line=22.5,
    side="over",
    bankroll=1000.0,
)

# Window before any fixture
_WINDOW_EMPTY = BacktestInput(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 1, 2),
    market="fouls_over_under",
    line=22.5,
    side="over",
    bankroll=1000.0,
)


def test_backtest_accumulates_results() -> None:
    bt = BacktestUseCase(predict=_build_predict_uc(), source=FakeTrainingDataSource())
    out = bt.execute(_WINDOW_M1)

    assert len(out.ev_results) > 0
    assert 0.0 <= out.hit_rate <= 1.0
    assert out.avg_nll > 0.0


def test_backtest_respects_date_window() -> None:
    bt = BacktestUseCase(predict=_build_predict_uc(), source=FakeTrainingDataSource())
    out = bt.execute(_WINDOW_EMPTY)

    assert len(out.ev_results) == 0
    assert out.hit_rate == 0.0
    assert out.avg_nll == 0.0
