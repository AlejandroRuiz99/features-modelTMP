"""FakeRefereeProfiler — pre-loaded in-memory profiles. Zero I/O."""

from __future__ import annotations

from HWFP.core.domain.exceptions import RefereeNotFoundError
from HWFP.core.domain.referee_profile import RefereeProfile


class FakeRefereeProfiler:
    """Looks up referee profiles from an in-memory dict.

    Happy-path fixture (R1): avg_fouls=24, std=3, sample_size=50.
    Raises RefereeNotFoundError for any unknown referee_id.
    """

    def __init__(self, profiles: dict[str, RefereeProfile] | None = None) -> None:
        self._profiles: dict[str, RefereeProfile] = (
            profiles if profiles is not None else {}
        )

    @classmethod
    def with_fixture(cls) -> FakeRefereeProfiler:
        """Return instance pre-loaded with golden e2e referee R1."""
        return cls(
            profiles={
                "R1": RefereeProfile(
                    referee_id="R1",
                    avg_fouls_per_match=24.0,
                    std_fouls_per_match=3.0,
                    sample_size=50,
                )
            }
        )

    def get_profile(self, referee_id: str) -> RefereeProfile:
        try:
            return self._profiles[referee_id]
        except KeyError:
            raise RefereeNotFoundError(f"Referee profile not found: {referee_id!r}")
