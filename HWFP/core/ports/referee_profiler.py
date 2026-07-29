"""RefereeProfiler port — fetch referee statistical profile."""

from __future__ import annotations

from typing import Protocol

from HWFP.core.domain.referee_profile import RefereeProfile


class RefereeProfiler(Protocol):
    """Look up a referee profile by ID.

    Raises:
        RefereeNotFoundError: If no profile exists for the given referee_id.
    """

    def get_profile(self, referee_id: str) -> RefereeProfile: ...
