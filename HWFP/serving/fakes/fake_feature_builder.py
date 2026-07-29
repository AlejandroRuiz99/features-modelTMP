"""FakeFeatureBuilder — deterministic fixed feature vector. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState

_FIXED_VECTOR: FeatureVector = (0.1, 0.2, 0.3, 0.4)


class FakeFeatureBuilder:
    """Returns a fixed 4-element feature vector for any input.

    Happy-path value for golden e2e: (0.1, 0.2, 0.3, 0.4).
    """

    def build(
        self, match: Match, home_state: TeamState, away_state: TeamState
    ) -> FeatureVector:
        return _FIXED_VECTOR
