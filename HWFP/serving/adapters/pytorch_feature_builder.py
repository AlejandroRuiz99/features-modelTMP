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

    `skip_market_fetch` is composition-controlled (default True — safe with
    no market data source wired) rather than hardcoded. Before this fix
    (Batch 6), `build()` always forwarded `skip_market_fetch=False`, which
    reached a legacy `from selection.odds_client import ...` call-time
    import that is not resolvable in production, crashing every real
    feature build. Set `skip_market_fetch=False` only after wiring
    `HWFP.features.assembly.betting_odds.set_market_data_source(...)` at
    composition time.
    """

    def __init__(
        self,
        state_provider_fn: Callable[[], dict[str, Any]],
        *,
        skip_market_fetch: bool = True,
    ) -> None:
        self._state_provider_fn = state_provider_fn
        self._skip_market_fetch = skip_market_fetch
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
            skip_market_fetch=self._skip_market_fetch,
            fecha_partido_input=match.kickoff.strftime("%Y-%m-%d"),
        )
        return tuple(float(flat_dict.get(k, 0.0)) for k in CANONICAL_FEATURE_KEYS)
