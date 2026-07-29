from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from HWFP.core.domain.exceptions import DomainValidationError


@dataclass(frozen=True)
class Match:
    match_id: str
    home_team_id: str
    away_team_id: str
    kickoff: datetime
    referee_id: str
    competition_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("match_id", self.match_id),
            ("home_team_id", self.home_team_id),
            ("away_team_id", self.away_team_id),
            ("referee_id", self.referee_id),
            ("competition_id", self.competition_id),
        ):
            if not value:
                raise DomainValidationError(f"{name} cannot be empty")
        if self.home_team_id == self.away_team_id:
            raise DomainValidationError("home and away teams must be different")
