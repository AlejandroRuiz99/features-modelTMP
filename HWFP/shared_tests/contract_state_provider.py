"""Contract tests for StateProvider port (REQ-8, REQ-9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.exceptions import StateNotFoundError
from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState

_MATCH_ID = "M1"
_HOME_TEAM_ID = "T_HOME"
_AS_OF = datetime(2026, 6, 16, 20, 0, 0)


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def state_provider(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_state_provider")
        return mod.FakeStateProvider.with_fixture()
    pytest.importorskip("HWFP.serving.adapters.supabase_state_adapter")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_get_match_returns_match(state_provider):
    result = state_provider.get_match(_MATCH_ID)
    assert isinstance(result, Match)
    assert result.match_id == _MATCH_ID


def test_get_team_state_returns_team_state(state_provider):
    result = state_provider.get_team_state(_HOME_TEAM_ID, _AS_OF)
    assert isinstance(result, TeamState)
    assert result.team_id == _HOME_TEAM_ID


def test_get_match_unknown_raises_state_not_found(state_provider):
    with pytest.raises(StateNotFoundError):
        state_provider.get_match("unknown-match-xyz-99")


def test_get_team_state_unknown_raises_state_not_found(state_provider):
    with pytest.raises(StateNotFoundError):
        state_provider.get_team_state("unknown-team-xyz-99", _AS_OF)


def test_state_provider_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.supabase_state_adapter")
    adapter = mod.SupabaseStateAdapter()
    with pytest.raises(NotImplementedError):
        adapter.get_match("any-match-id")
