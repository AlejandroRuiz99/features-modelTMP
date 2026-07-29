"""LineupProvider port — retrieve confirmed match lineups."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from HWFP.core.domain.lineup import Lineup


@runtime_checkable
class LineupProvider(Protocol):
    """Retrieve confirmed starting lineups for a match."""

    def get_lineup(self, match_id: str) -> Lineup | None: ...
