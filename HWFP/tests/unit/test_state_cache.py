"""Unit tests for HWFP.features.core.state_cache — absorbed state cache.

TDD cycle:
  RED   — set_data_source()/get_state() fail: DI hook did not exist yet on
          the legacy singleton-Supabase-only implementation
  GREEN — get_state() is a zero-arg callable (design D1's state_provider_fn
          contract) backed by an injectable data source, returning the
          shape required by scenario "State shape"
"""

from __future__ import annotations

import pytest

from HWFP.features.core import state_cache


@pytest.fixture(autouse=True)
def _reset_state_cache():
    """Isolate the module-level singleton cache and data source between tests."""
    state_cache._cached = None
    state_cache._data_source = None
    yield
    state_cache._cached = None
    state_cache._data_source = None


_REQUIRED_KEYS = {"partidos", "xstyles", "scores", "ref_perfiles", "perfiles_gmm", "updated_at"}


def _stub_partidos() -> list[dict]:
    return [
        {
            "date": "2025-08-16",
            "season": 2025,
            "referee": "Test Referee",
            "home": {"name": "Real Madrid", "fouls": 9, "yellow_cards": 1, "red_cards": 0, "goals": 2},
            "away": {"name": "Barcelona", "fouls": 10, "yellow_cards": 2, "red_cards": 0, "goals": 1},
        }
    ]


class TestGetStateShape:
    """Scenario 'State shape': GIVEN a stubbed data source, get_state() returns every required key."""

    def test_get_state_is_zero_arg_callable(self) -> None:
        """design D1: state_provider_fn must be invocable with zero arguments."""
        state_cache.set_data_source(_stub_partidos)
        result = state_cache.get_state()
        assert isinstance(result, dict)

    def test_get_state_contains_all_required_keys(self) -> None:
        state_cache.set_data_source(_stub_partidos)
        result = state_cache.get_state()
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_get_state_partidos_matches_data_source(self) -> None:
        state_cache.set_data_source(_stub_partidos)
        result = state_cache.get_state()
        assert result["partidos"] == _stub_partidos()

    def test_get_state_derives_scores_and_xstyles_from_partidos(self) -> None:
        """Not just present -- actually computed from the injected data (not empty stubs)."""
        state_cache.set_data_source(_stub_partidos)
        result = state_cache.get_state()
        assert "Real Madrid" in result["scores"]
        assert "Barcelona" in result["xstyles"]


class TestGetStateWithoutDataSource:
    def test_raises_clear_error_when_no_data_source_configured(self) -> None:
        with pytest.raises(RuntimeError, match="data source"):
            state_cache.get_state()


class TestGetStateCaching:
    def test_second_call_does_not_re_invoke_data_source(self) -> None:
        calls = {"n": 0}

        def _counting_source() -> list[dict]:
            calls["n"] += 1
            return _stub_partidos()

        state_cache.set_data_source(_counting_source)
        state_cache.get_state()
        state_cache.get_state()
        assert calls["n"] == 1

    def test_refresh_true_re_invokes_data_source(self) -> None:
        calls = {"n": 0}

        def _counting_source() -> list[dict]:
            calls["n"] += 1
            return _stub_partidos()

        state_cache.set_data_source(_counting_source)
        state_cache.get_state()
        state_cache.get_state(refresh=True)
        assert calls["n"] == 2
