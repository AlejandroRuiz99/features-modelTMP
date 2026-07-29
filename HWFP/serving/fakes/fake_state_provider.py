"""FakeStateProvider — pre-loaded in-memory match and team state. Zero I/O."""

from __future__ import annotations

from datetime import datetime

from HWFP.core.domain.exceptions import StateNotFoundError
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState

_KICKOFF = datetime(2026, 6, 16, 20, 0, 0)
_MATCH_FIXTURE = Match(
    match_id="M1",
    home_team_id="T_HOME",
    away_team_id="T_AWAY",
    kickoff=_KICKOFF,
    referee_id="R1",
    competition_id="LA_LIGA",
)
_TEAM_STATES_FIXTURE: dict[str, TeamState] = {
    "T_HOME": TeamState(
        team_id="T_HOME",
        as_of=_KICKOFF,
        avg_fouls_per_match=12.5,
        avg_fouls_conceded=11.0,
        form_window=5,
    ),
    "T_AWAY": TeamState(
        team_id="T_AWAY",
        as_of=_KICKOFF,
        avg_fouls_per_match=11.5,
        avg_fouls_conceded=12.0,
        form_window=5,
    ),
}


class FakeStateProvider:
    """Looks up match and team state from in-memory dicts.

    Happy-path fixture (M1): home=T_HOME, away=T_AWAY, referee=R1, kickoff=2026-06-16T20:00.
    Raises StateNotFoundError for any unknown match_id or team_id.
    """

    def __init__(
        self,
        matches: dict[str, Match] | None = None,
        team_states: dict[str, TeamState] | None = None,
    ) -> None:
        self._matches: dict[str, Match] = matches if matches is not None else {}
        self._team_states: dict[str, TeamState] = (
            team_states if team_states is not None else {}
        )

    @classmethod
    def with_fixture(cls) -> FakeStateProvider:
        """Return instance pre-loaded with golden e2e match M1 and team states."""
        return cls(
            matches={"M1": _MATCH_FIXTURE},
            team_states=_TEAM_STATES_FIXTURE,
        )

    def get_match(self, match_id: str) -> Match:
        try:
            return self._matches[match_id]
        except KeyError:
            raise StateNotFoundError(f"Match not found: {match_id!r}")

    def get_team_state(self, team_id: str, as_of: datetime) -> TeamState:
        try:
            return self._team_states[team_id]
        except KeyError:
            raise StateNotFoundError(f"Team state not found: {team_id!r}")
