"""Contract tests for FeatureBuilder port (REQ-8, REQ-9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.match import Match
from HWFP.core.domain.team_state import TeamState


def _match() -> Match:
    return Match(
        match_id="M1",
        home_team_id="T_HOME",
        away_team_id="T_AWAY",
        kickoff=datetime(2026, 6, 16, 20, 0, 0),
        referee_id="R1",
        competition_id="la-liga",
    )


def _team_state(team_id: str) -> TeamState:
    return TeamState(
        team_id=team_id,
        as_of=datetime(2026, 6, 16, 20, 0, 0),
        avg_fouls_per_match=12.5,
        avg_fouls_conceded=11.0,
        form_window=5,
    )


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def feature_builder(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_feature_builder")
        return mod.FakeFeatureBuilder()
    pytest.importorskip("HWFP.serving.adapters.pytorch_feature_builder")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_build_returns_tuple_of_floats(feature_builder):
    result = feature_builder.build(
        _match(), _team_state("T_HOME"), _team_state("T_AWAY")
    )
    assert isinstance(result, tuple)
    assert all(isinstance(v, float) for v in result)


def test_build_is_deterministic(feature_builder):
    m = _match()
    home = _team_state("T_HOME")
    away = _team_state("T_AWAY")
    assert feature_builder.build(m, home, away) == feature_builder.build(m, home, away)


def test_build_produces_non_empty_vector(feature_builder):
    result = feature_builder.build(
        _match(), _team_state("T_HOME"), _team_state("T_AWAY")
    )
    assert len(result) > 0


def test_feature_builder_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.pytorch_feature_builder")
    adapter = mod.PyTorchFeatureBuilder()
    with pytest.raises(NotImplementedError):
        adapter.build(_match(), _team_state("T_HOME"), _team_state("T_AWAY"))
