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

    Default-fill contract: `build_features()` returns a raw dict of 78 keys,
    not the 76 `CANONICAL_FEATURE_KEYS`. 5 canonical keys
    (`referee_avg_fouls`, `referee_home_bias`, `referee_is_shrunk`,
    `referee_team_committed_home`, `referee_team_committed_away`) are absent
    from the raw output and are default-filled to `0.0` below via
    `flat_dict.get(k, 0.0)`. 7 raw keys are string/metadata fields
    (`home_team`, `away_team`, `date`, `season`, `referee`,
    `intensidad_esperada`, `riesgo_disciplinario`) that carry no canonical
    numeric feature and are silently dropped by the projection onto
    `CANONICAL_FEATURE_KEYS`. This is byte-identical to legacy behavior, not
    a regression — see `HWFP.core.domain.feature_keys` module docstring and
    `tests/unit/test_feature_golden_vector.py::test_key_set_matches_canonical_feature_keys`
    for the pinned, regression-guarded contract.
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
