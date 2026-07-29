"""PredictMatchUseCase — 13-step orchestration of the foul prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from HWFP.core.domain.ev_result import EVResult
from HWFP.core.domain.match_prediction import MatchPrediction
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_status import ModelStatus
from HWFP.core.domain.stake_result import StakeResult
from HWFP.core.ports.ev_calculator import EVCalculator
from HWFP.core.ports.feature_builder import FeatureBuilder
from HWFP.core.ports.model_registry import ModelRegistry
from HWFP.core.ports.odds_provider import OddsProvider
from HWFP.core.ports.overlay_engine import OverlayEngine
from HWFP.core.ports.prediction_sink import PredictionSink
from HWFP.core.ports.referee_profiler import RefereeProfiler
from HWFP.core.ports.staking_calculator import StakingCalculator
from HWFP.core.ports.state_provider import StateProvider


@dataclass
class PredictMatchInput:
    match_id: str
    market: str
    line: float
    side: str
    bankroll: float


@dataclass
class PredictMatchOutput:
    prediction: MatchPrediction
    ev: EVResult
    stake: StakeResult


class PredictMatchUseCase:
    def __init__(
        self,
        states: StateProvider,
        odds: OddsProvider,
        features: FeatureBuilder,
        model_registry: ModelRegistry,
        referee: RefereeProfiler,
        overlay: OverlayEngine,
        ev_calc: EVCalculator,
        staking: StakingCalculator,
        sink: PredictionSink,
        clock: Callable[[], datetime],
    ) -> None:
        self._states = states
        self._odds = odds
        self._features = features
        self._model_registry = model_registry
        self._referee = referee
        self._overlay = overlay
        self._ev_calc = ev_calc
        self._staking = staking
        self._sink = sink
        self._clock = clock

    def execute(self, inp: PredictMatchInput) -> PredictMatchOutput:
        # 1. Fetch match
        match = self._states.get_match(inp.match_id)
        # 2-3. Fetch team states as of kickoff
        home = self._states.get_team_state(match.home_team_id, match.kickoff)
        away = self._states.get_team_state(match.away_team_id, match.kickoff)
        # 4. Fetch referee profile (available for enriched feature builders)
        self._referee.get_profile(match.referee_id)
        # 5. Build feature vector
        fvec = self._features.build(match, home, away)
        # 6. Load production model; resolve its manifest model_id
        model = self._model_registry.load_production()
        prod_manifests = [
            m
            for m in self._model_registry.list_manifests()
            if m.status == ModelStatus.PRODUCTION
        ]
        model_id = (
            prod_manifests[0].model_id if prod_manifests else ModelId("production")
        )
        # 7. Predict PMF
        pmf = model.predict(fvec)
        # 8. Fetch market odds
        odds_q = self._odds.get_odds(inp.match_id, inp.market)
        # 9. Compute expected value
        ev = self._ev_calc.compute(pmf, odds_q, inp.line)
        # 10. Compute stake
        stake = self._staking.compute(ev, inp.bankroll)
        # 11. Build prediction record
        prediction = MatchPrediction(
            match_id=inp.match_id,
            pmf=pmf,
            model_id=model_id,
            generated_at=self._clock(),
        )
        # 12. Persist
        self._sink.write(prediction)
        # 13. Return
        return PredictMatchOutput(prediction=prediction, ev=ev, stake=stake)
