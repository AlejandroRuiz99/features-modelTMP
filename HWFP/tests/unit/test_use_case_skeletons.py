"""B3 — Use Case Skeleton tests (REQ-3).

RED phase: all tests fail with ImportError until use case modules are created.
GREEN phase: all 5 use cases importable, init params correct, execute() raises NotImplementedError.
"""

from __future__ import annotations

import inspect

from HWFP.core.application.backtest import (
    BacktestInput,
    BacktestOutput,
    BacktestUseCase,
)
from HWFP.core.application.predict_match import (
    PredictMatchInput,
    PredictMatchOutput,
    PredictMatchUseCase,
)
from HWFP.core.application.promote_model import PromoteModelUseCase, PromotionGates
from HWFP.core.application.simulate_staking import (
    SimulateInput,
    SimulateOutput,
    SimulateStakingUseCase,
)
from HWFP.core.application.train_model import TrainInput, TrainModelUseCase


# ---------------------------------------------------------------------------
# T3.1 — PredictMatchUseCase constructor signature
# ---------------------------------------------------------------------------


def test_predict_match_init_params() -> None:
    """REQ-3: __init__ must accept exactly the 10 documented params (9 ports + clock)."""
    sig = inspect.signature(PredictMatchUseCase.__init__)
    params = set(sig.parameters) - {"self"}
    expected = {
        "states",
        "odds",
        "features",
        "model_registry",
        "referee",
        "overlay",
        "ev_calc",
        "staking",
        "sink",
        "clock",
    }
    assert params == expected


def test_predict_match_execute_param_is_inp() -> None:
    """REQ-3: execute() must accept a single positional param named 'inp'."""
    sig = inspect.signature(PredictMatchUseCase.execute)
    assert "inp" in sig.parameters


# ---------------------------------------------------------------------------
# T3.1 — other use-case constructor params (smoke: at least documented names present)
# ---------------------------------------------------------------------------


def test_backtest_init_params() -> None:
    sig = inspect.signature(BacktestUseCase.__init__)
    params = set(sig.parameters) - {"self"}
    assert {"predict", "source"} == params


def test_simulate_staking_init_params() -> None:
    sig = inspect.signature(SimulateStakingUseCase.__init__)
    params = set(sig.parameters) - {"self"}
    assert {"staking"} == params


def test_train_model_init_params() -> None:
    sig = inspect.signature(TrainModelUseCase.__init__)
    params = set(sig.parameters) - {"self"}
    assert {"source", "trainer", "registry", "clock", "git_sha_provider"} == params


def test_promote_model_init_params() -> None:
    sig = inspect.signature(PromoteModelUseCase.__init__)
    params = set(sig.parameters) - {"self"}
    assert {"registry", "gates"} == params


# ---------------------------------------------------------------------------
# T3.1 — I/O dataclasses importable
# ---------------------------------------------------------------------------


def test_io_dataclasses_importable() -> None:
    """All Input/Output dataclasses must be importable (locked signatures)."""
    for cls in (
        PredictMatchInput,
        PredictMatchOutput,
        BacktestInput,
        BacktestOutput,
        SimulateInput,
        SimulateOutput,
        TrainInput,
    ):
        assert cls is not None


# ---------------------------------------------------------------------------
# T3.1 — importable from HWFP.core.application (re-export __init__)
# ---------------------------------------------------------------------------


def test_use_cases_importable_from_application_package() -> None:
    """All 5 use cases must be importable directly from HWFP.core.application."""
    import HWFP.core.application as app

    for name in (
        "PredictMatchUseCase",
        "BacktestUseCase",
        "SimulateStakingUseCase",
        "TrainModelUseCase",
        "PromoteModelUseCase",
    ):
        assert hasattr(app, name), f"HWFP.core.application missing: {name}"
