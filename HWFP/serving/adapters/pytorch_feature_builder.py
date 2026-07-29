"""PyTorchFeatureBuilder — FeatureBuilder port adapter wrapping build_features()."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState
from HWFP.serving.adapters._feature_keys import CANONICAL_FEATURE_KEYS

_REPO_ROOT = Path(__file__).parents[3]


def _ensure_legacy_paths() -> None:
    p = str(_REPO_ROOT / "features_generator")
    if p not in sys.path:
        sys.path.insert(0, p)


class PyTorchFeatureBuilder:
    """Bridges domain objects → legacy build_features() → FeatureVector.

    The state_provider_fn must return the full state dict consumed by the
    legacy pipeline (keys: partidos, xstyles, scores, ref_perfiles, ...).
    This callable is invoked on every build() call to ensure freshness.
    """

    def __init__(self, state_provider_fn: Callable[[], dict[str, Any]]) -> None:
        self._state_provider_fn = state_provider_fn
        _ensure_legacy_paths()
        from assembly.feature_assembler import build_features

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
