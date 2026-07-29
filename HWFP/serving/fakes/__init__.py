"""Serving fakes — deterministic in-memory implementations for all serving ports."""

from __future__ import annotations

from HWFP.serving.fakes.fake_ev_calculator import FakeEVCalculator
from HWFP.serving.fakes.fake_feature_builder import FakeFeatureBuilder
from HWFP.serving.fakes.fake_foul_model import FakeFoulModel
from HWFP.serving.fakes.fake_model_registry import FakeModelRegistry
from HWFP.serving.fakes.fake_odds_provider import FakeOddsProvider
from HWFP.serving.fakes.fake_overlay_engine import FakeOverlayEngine
from HWFP.serving.fakes.fake_prediction_sink import FakePredictionSink
from HWFP.serving.fakes.fake_referee_profiler import FakeRefereeProfiler
from HWFP.serving.fakes.fake_staking_calculator import FakeStakingCalculator
from HWFP.serving.fakes.fake_state_provider import FakeStateProvider

__all__ = [
    "FakeEVCalculator",
    "FakeFeatureBuilder",
    "FakeFoulModel",
    "FakeModelRegistry",
    "FakeOddsProvider",
    "FakeOverlayEngine",
    "FakePredictionSink",
    "FakeRefereeProfiler",
    "FakeStakingCalculator",
    "FakeStateProvider",
]
