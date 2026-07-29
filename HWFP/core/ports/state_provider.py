"""StateProvider port — read match and team state."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState


class StateProvider(Protocol):
    """Read-only access to match and team state.

    Raises:
        StateNotFoundError: If the requested match or team is not found.
    """

    def get_match(self, match_id: str) -> Match: ...

    def get_team_state(self, team_id: str, as_of: datetime) -> TeamState: ...
