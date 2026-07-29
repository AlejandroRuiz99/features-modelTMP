"""Serving composition root — pure factory functions, no global state."""

from __future__ import annotations

from datetime import datetime

from HWFP.core.application.predict_match import PredictMatchUseCase
from HWFP.serving.adapters.aggregated_odds_provider import AggregatedOddsProvider
from HWFP.serving.adapters.cgm_odds_stub import CGMOddsStub
from HWFP.serving.adapters.codere_odds_stub import CodereOddsStub
from HWFP.serving.adapters.onexbet_odds_stub import OnexBetOddsStub
from HWFP.serving.fakes import (
    FakeEVCalculator,
    FakeFeatureBuilder,
    FakeModelRegistry,
    FakeOddsProvider,
    FakeOverlayEngine,
    FakePredictionSink,
    FakeRefereeProfiler,
    FakeStakingCalculator,
    FakeStateProvider,
)


def container_fakes() -> PredictMatchUseCase:
    """Wire PredictMatchUseCase with all fakes. Deterministic. Zero I/O."""
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


def container_production() -> PredictMatchUseCase:
    """Raises until real adapters are implemented."""
    raise NotImplementedError(
        "Production adapters are stubs. Implement in: "
        "hwfp-supabase-state-adapter, hwfp-codere-odds-adapter, "
        "hwfp-pytorch-foul-model-adapter."
    )


def container_stubs() -> dict:
    """All-stubs odds container for development without real data sources."""
    odds = AggregatedOddsProvider([
        CodereOddsStub(),
        CGMOddsStub(),
        OnexBetOddsStub(),
    ])
    return {"odds_provider": odds}
