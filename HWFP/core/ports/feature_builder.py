"""FeatureBuilder port — assemble feature vector from match context."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.feature_vector import FeatureVector
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState


class FeatureBuilder(Protocol):
    """Assemble a FeatureVector from match and team state.

    Pure function — no I/O, no side effects. Deterministic.
    """

    def build(
        self, match: Match, home_state: TeamState, away_state: TeamState
    ) -> FeatureVector: ...
