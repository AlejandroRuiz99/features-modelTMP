"""PyTorchFeatureBuilder — FeatureBuilder port adapter wrapping build_features()."""

from __future__ import annotations

from typing import Any, Callable

from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState
from HWFP.features.assembly.feature_assembler import build_features
from HWFP.serving.adapters._feature_keys import CANONICAL_FEATURE_KEYS


class PyTorchFeatureBuilder:
    """Bridges domain objects → HWFP.features.build_features() → FeatureVector.

    The state_provider_fn must return the full state dict consumed by the
    absorbed feature pipeline (keys: partidos, xstyles, scores, ref_perfiles,
    ...). This callable is invoked on every build() call to ensure freshness
    (design D1: composition injects a zero-arg state_provider_fn backed by
    HWFP.features.core.state_cache.get_state).
    """

    def __init__(self, state_provider_fn: Callable[[], dict[str, Any]]) -> None:
        self._state_provider_fn = state_provider_fn
        self._build_features = build_features

    def build(
        self, match: Match, home_state: TeamState, away_state: TeamState
    ) -> FeatureVector:
        state = self._state_provider_fn()
        flat_dict = self._build_features(
            state=state,
            equipo_local_input=match.home_team_id,
            equipo_visitante_input=match.away_team_id,
            arbitro_input=match.referee_id,
            jornada=None,
            skip_market_fetch=False,
            fecha_partido_input=match.kickoff.strftime("%Y-%m-%d"),
        )
        return tuple(float(flat_dict.get(k, 0.0)) for k in CANONICAL_FEATURE_KEYS)
