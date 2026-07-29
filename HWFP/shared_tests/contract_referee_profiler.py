"""Contract tests for RefereeProfiler port (REQ-8, REQ-9)."""

from __future__ import annotations

import pytest

from HWFP.core.domain.exceptions import RefereeNotFoundError
from HWFP.core.domain.referee_profile import RefereeProfile

_REFEREE_ID = "R1"


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def referee_profiler(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_referee_profiler")
        return mod.FakeRefereeProfiler.with_fixture()
    pytest.importorskip("HWFP.serving.adapters.yaml_referee_profiler")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_get_profile_returns_referee_profile(referee_profiler):
    result = referee_profiler.get_profile(_REFEREE_ID)
    assert isinstance(result, RefereeProfile)
    assert result.referee_id == _REFEREE_ID


def test_get_profile_unknown_raises_referee_not_found(referee_profiler):
    with pytest.raises(RefereeNotFoundError):
        referee_profiler.get_profile("unknown-referee-xyz-99")


def test_referee_profiler_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.yaml_referee_profiler")
    adapter = mod.YamlRefereeProfiler()
    with pytest.raises(NotImplementedError):
        adapter.get_profile("R1")
