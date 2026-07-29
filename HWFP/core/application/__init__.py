"""HWFP core application layer — use case skeletons (B3)."""

from __future__ import annotations

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

__all__ = [
    "BacktestInput",
    "BacktestOutput",
    "BacktestUseCase",
    "PredictMatchInput",
    "PredictMatchOutput",
    "PredictMatchUseCase",
    "PromoteModelUseCase",
    "PromotionGates",
    "SimulateInput",
    "SimulateOutput",
    "SimulateStakingUseCase",
    "TrainInput",
    "TrainModelUseCase",
]
