"""Adapter stub — YamlRefereeProfiler (port: RefereeProfiler)."""

from __future__ import annotations

from HWFP.core.domain.referee_profile import RefereeProfile


class YamlRefereeProfiler:
    """Stub for RefereeProfiler. Raises NotImplementedError on all methods."""

    def get_profile(self, referee_id: str) -> RefereeProfile:
        raise NotImplementedError(
            "HWFP adapter stub — port=RefereeProfiler; future change hwfp-yaml-referee-profiler-adapter"
        )
