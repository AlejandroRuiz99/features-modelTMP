"""T8.2 — Container wiring: fakes vs production stubs."""

from __future__ import annotations

import pytest

from HWFP.core.application.predict_match import PredictMatchUseCase
from HWFP.serving.composition.container import container_fakes, container_production
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


def test_container_fakes_returns_predict_use_case() -> None:
    use_case = container_fakes()
    assert isinstance(use_case, PredictMatchUseCase)


def test_container_fakes_ports_are_fakes() -> None:
    use_case = container_fakes()
    assert isinstance(use_case._states, FakeStateProvider)
    assert isinstance(use_case._odds, FakeOddsProvider)
    assert isinstance(use_case._features, FakeFeatureBuilder)
    assert isinstance(use_case._model_registry, FakeModelRegistry)
    assert isinstance(use_case._referee, FakeRefereeProfiler)
    assert isinstance(use_case._overlay, FakeOverlayEngine)
    assert isinstance(use_case._ev_calc, FakeEVCalculator)
    assert isinstance(use_case._staking, FakeStakingCalculator)
    assert isinstance(use_case._sink, FakePredictionSink)


def test_container_production_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        container_production()
