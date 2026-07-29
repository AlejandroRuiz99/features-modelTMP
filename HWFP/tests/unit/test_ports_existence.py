"""B2 — Port existence and Protocol compliance tests (T2.1).

RED: fails with ImportError until all 12 ports are created.
GREEN: passes once ports are defined in HWFP/core/ports/.
"""

from __future__ import annotations

import inspect
from typing import Protocol

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _is_protocol(cls: type) -> bool:
    return getattr(cls, "_is_protocol", False)


def _public_methods(cls: type) -> set:
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def _param_names(cls: type, method: str) -> tuple:
    sig = inspect.signature(getattr(cls, method))
    return tuple(p for p in sig.parameters if p != "self")


# ── imports (all 12 ports) ────────────────────────────────────────────────────

from HWFP.core.ports import (  # noqa: E402
    EVCalculator,
    FeatureBuilder,
    FoulModel,
    ModelRegistry,
    ModelTrainer,
    OddsProvider,
    OverlayEngine,
    PredictionSink,
    RefereeProfiler,
    StakingCalculator,
    StateProvider,
    TrainingDataSource,
)

_ALL_PORTS = [
    StateProvider,
    OddsProvider,
    FeatureBuilder,
    FoulModel,
    RefereeProfiler,
    OverlayEngine,
    EVCalculator,
    StakingCalculator,
    PredictionSink,
    ModelRegistry,
    TrainingDataSource,
    ModelTrainer,
]

# ── T2.1a — each port is a typing.Protocol ───────────────────────────────────


@pytest.mark.parametrize("port_cls", _ALL_PORTS, ids=[c.__name__ for c in _ALL_PORTS])
def test_each_port_is_a_protocol(port_cls: type) -> None:
    assert _is_protocol(port_cls), f"{port_cls.__name__} must be typing.Protocol"


# ── T2.1b — each port declares the documented methods ────────────────────────


@pytest.mark.parametrize(
    "port_cls, expected_methods",
    [
        (StateProvider, {"get_match", "get_team_state"}),
        (OddsProvider, {"get_odds"}),
        (FeatureBuilder, {"build"}),
        (FoulModel, {"predict"}),
        (RefereeProfiler, {"get_profile"}),
        (OverlayEngine, {"compute"}),
        (EVCalculator, {"compute"}),
        (StakingCalculator, {"compute"}),
        (PredictionSink, {"write"}),
        (
            ModelRegistry,
            {"load_production", "register", "promote", "archive", "list_manifests"},
        ),
        (TrainingDataSource, {"iter_examples", "dataset_hash"}),
        (ModelTrainer, {"fit"}),
    ],
    ids=[
        "StateProvider",
        "OddsProvider",
        "FeatureBuilder",
        "FoulModel",
        "RefereeProfiler",
        "OverlayEngine",
        "EVCalculator",
        "StakingCalculator",
        "PredictionSink",
        "ModelRegistry",
        "TrainingDataSource",
        "ModelTrainer",
    ],
)
def test_port_has_declared_methods(port_cls: type, expected_methods: set) -> None:
    actual = _public_methods(port_cls)
    missing = expected_methods - actual
    assert not missing, f"{port_cls.__name__} is missing methods: {missing}"


# ── T2.1c — method parameter names match the design ──────────────────────────


def test_state_provider_get_match_params() -> None:
    assert _param_names(StateProvider, "get_match") == ("match_id",)


def test_state_provider_get_team_state_params() -> None:
    assert _param_names(StateProvider, "get_team_state") == ("team_id", "as_of")


def test_odds_provider_get_odds_params() -> None:
    assert _param_names(OddsProvider, "get_odds") == ("match_id", "market")


def test_feature_builder_build_params() -> None:
    assert _param_names(FeatureBuilder, "build") == (
        "match",
        "home_state",
        "away_state",
    )


def test_foul_model_predict_params() -> None:
    assert _param_names(FoulModel, "predict") == ("features",)


def test_referee_profiler_get_profile_params() -> None:
    assert _param_names(RefereeProfiler, "get_profile") == ("referee_id",)


def test_overlay_engine_compute_params() -> None:
    assert _param_names(OverlayEngine, "compute") == ("pmf", "odds")


def test_ev_calculator_compute_params() -> None:
    assert _param_names(EVCalculator, "compute") == ("pmf", "odds", "line")


def test_staking_calculator_compute_params() -> None:
    assert _param_names(StakingCalculator, "compute") == ("ev", "bankroll")


def test_prediction_sink_write_params() -> None:
    assert _param_names(PredictionSink, "write") == ("prediction",)


def test_model_registry_load_production_no_params() -> None:
    assert _param_names(ModelRegistry, "load_production") == ()


def test_model_registry_register_params() -> None:
    assert _param_names(ModelRegistry, "register") == ("manifest", "model_blob")


def test_model_registry_promote_params() -> None:
    assert _param_names(ModelRegistry, "promote") == ("model_id",)


def test_model_registry_archive_params() -> None:
    assert _param_names(ModelRegistry, "archive") == ("model_id",)


def test_model_registry_list_manifests_no_params() -> None:
    assert _param_names(ModelRegistry, "list_manifests") == ()


def test_training_data_source_iter_examples_no_params() -> None:
    assert _param_names(TrainingDataSource, "iter_examples") == ()


def test_training_data_source_dataset_hash_no_params() -> None:
    assert _param_names(TrainingDataSource, "dataset_hash") == ()


def test_model_trainer_fit_params() -> None:
    assert _param_names(ModelTrainer, "fit") == ("examples", "hyperparams")
