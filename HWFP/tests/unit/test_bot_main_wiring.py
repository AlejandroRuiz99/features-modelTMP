"""Unit tests for HWFP.cli.bot_main — composition-root wiring (Batch 4, PR4).

Spec capability `serving-composition` (MODIFIED):
  Scenario "Import succeeds"  — `import HWFP.cli.bot_main` raises no ImportError.
  Scenario "Wiring test"      — `build_container()` with injected fakes returns
                                 a container, and a feature-build + predict
                                 round trip succeeds.

TDD cycle:
  RED   — before this batch, `main()` had 3 wiring defects that made the
          equivalent code path crash: `PytorchFeatureBuilder` (typo, does not
          exist), `FilesystemModelRegistry()` (missing required
          `checkpoints_dir`), and `state_provider_fn=states.get_team_state`
          (a 2-arg method passed where `PyTorchFeatureBuilder` needs a
          zero-arg callable). None of this was testable because everything
          lived inline inside `main()`. This file's tests fail before
          `build_container()` exists (ImportError) and would fail against
          the old wiring if re-pointed at it (AttributeError / TypeError).
  GREEN — `build_container()` is extracted as a pure factory (no Telegram or
          env access — all credentials/config come in as parameters) with
          the 3 fixes applied, and calls
          `HWFP.features.core.state_cache.set_data_source(...)` at
          composition time (B2 deviation #1 — required alongside wiring
          `state_provider_fn=state_cache.get_state`, since a zero-arg
          `get_state` with no configured data source raises RuntimeError).

Fakes only: this test never exercises the real `HWFP.features` pipeline
(`PyTorchFeatureBuilder`/`build_features`) end-to-end against a live market
data source. `PyTorchFeatureBuilder` now defaults `skip_market_fetch=True`
(Batch 6 fix — see `test_model_adapters.py::TestPyTorchFeatureBuilder` and
`HWFP.features.assembly.betting_odds.set_market_data_source`), so real
feature builds no longer crash; a real Supabase-backed market-odds fetcher
still has not been wired at composition time (tracked alongside the
Supabase env-var gap below). Using `FakeFeatureBuilder` + `FakeModelRegistry`
keeps this test hermetic and fast while still proving `build_container()`'s
wiring is correct end-to-end.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.features.core import state_cache


@pytest.fixture(autouse=True)
def _reset_state_cache():
    """Isolate the module-level singleton cache/data-source between tests."""
    state_cache._cached = None
    state_cache._data_source = None
    yield
    state_cache._cached = None
    state_cache._data_source = None


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


def _fake_overrides() -> dict:
    """Fakes for every port `build_container()` accepts, PLUS `bot`/
    `scheduler` placeholders.

    `bot`/`scheduler`: the real `TelegramBotRunner`/`SchedulerRunner`
    require the optional `python-telegram-bot`/`apscheduler` packages, which
    this test environment does not have installed (the same pre-existing,
    out-of-scope gap documented for `test_telegram_formatters.py`'s
    collection error) — injecting placeholders proves build_container()'s
    own wiring without depending on those packages being present.

    `states`/`tracker`: the real `SupabaseStateAdapter`/`SupabasePerformanceTracker`
    require `url`/`key` constructor args that `bot_main.py` has never read
    from the environment (a pre-existing gap predating this batch's 3 named
    defects — see apply-progress Issues Found). Not exercised by any
    assertion in this file, so lightweight placeholders are sufficient.
    """
    from HWFP.serving.fakes import FakeFeatureBuilder, FakeModelRegistry, FakeStateProvider

    return {
        "feature_builder": FakeFeatureBuilder(),
        "model_registry": FakeModelRegistry.with_production_model(),
        "partidos_source": _stub_partidos,
        "states": FakeStateProvider.with_fixture(),
        "tracker": object(),
        "bot": object(),
        "scheduler": object(),
    }


# ---------------------------------------------------------------------------
# Scenario "Import succeeds"
# ---------------------------------------------------------------------------


def test_bot_main_imports_cleanly() -> None:
    import HWFP.cli.bot_main  # noqa: F401


# ---------------------------------------------------------------------------
# Scenario "Wiring test"
# ---------------------------------------------------------------------------


class TestBuildContainer:
    def test_returns_a_container(self) -> None:
        from HWFP.cli.bot_main import build_container

        container = build_container(token="fake-token", chat_id=1, **_fake_overrides())
        assert container is not None

    def test_wires_state_cache_data_source_at_composition_time(self) -> None:
        """B2 deviation #1: build_container() must call set_data_source(),
        not just wire state_provider_fn=get_state — otherwise the very first
        get_state() call raises RuntimeError (no data source configured).
        """
        from HWFP.cli.bot_main import build_container

        build_container(token="fake-token", chat_id=1, **_fake_overrides())

        result = state_cache.get_state()  # must not raise
        assert result["partidos"] == _stub_partidos()

    def test_feature_build_and_predict_round_trip_succeeds(self) -> None:
        from HWFP.core.domain.foul_pmf import FoulPMF
        from HWFP.core.domain.match import Match
        from HWFP.core.domain.team_state import TeamState
        from HWFP.cli.bot_main import build_container

        container = build_container(token="fake-token", chat_id=1, **_fake_overrides())

        match = Match(
            match_id="m1",
            home_team_id="Real Madrid",
            away_team_id="Barcelona",
            kickoff=datetime(2025, 11, 10, 20, 0),
            referee_id="Test Referee",
            competition_id="laliga",
        )
        home_state = TeamState(
            team_id="Real Madrid", as_of=datetime(2025, 11, 10),
            avg_fouls_per_match=12.0, avg_fouls_conceded=10.5, form_window=5,
        )
        away_state = TeamState(
            team_id="Barcelona", as_of=datetime(2025, 11, 10),
            avg_fouls_per_match=11.0, avg_fouls_conceded=9.5, form_window=5,
        )

        features = container.feature_builder.build(match, home_state, away_state)
        model = container.model_registry.load_production()
        pmf = model.predict(features)

        assert isinstance(pmf, FoulPMF)


class TestBuildContainerNamedDefectFixes:
    """Each test targets exactly one of the 3 verified bot_main.py defects."""

    def test_default_feature_builder_is_pytorch_feature_builder_not_typo(self) -> None:
        """Defect 1: `PytorchFeatureBuilder` (typo) does not exist; the
        default feature_builder must be the real `PyTorchFeatureBuilder`.
        """
        from HWFP.cli.bot_main import build_container
        from HWFP.serving.adapters.pytorch_feature_builder import PyTorchFeatureBuilder
        from HWFP.serving.fakes import FakeModelRegistry

        container = build_container(
            token="fake-token",
            chat_id=1,
            model_registry=FakeModelRegistry.with_production_model(),
            partidos_source=_stub_partidos,
            states=object(),
            tracker=object(),
            bot=object(),
            scheduler=object(),
        )
        assert isinstance(container.feature_builder, PyTorchFeatureBuilder)

    def test_default_feature_builder_state_provider_fn_is_zero_arg_state_cache(
        self,
    ) -> None:
        """Defect 3: `state_provider_fn` must be the zero-arg
        `state_cache.get_state`, not the 2-arg `states.get_team_state`.
        """
        from HWFP.cli.bot_main import build_container
        from HWFP.serving.fakes import FakeModelRegistry

        container = build_container(
            token="fake-token",
            chat_id=1,
            model_registry=FakeModelRegistry.with_production_model(),
            partidos_source=_stub_partidos,
            states=object(),
            tracker=object(),
            bot=object(),
            scheduler=object(),
        )
        # A 2-arg bound method (states.get_team_state) would raise
        # TypeError when called with zero arguments; state_cache.get_state
        # accepts zero positional args (refresh is keyword-only).
        result = container.feature_builder._state_provider_fn()
        assert isinstance(result, dict)

    def test_default_model_registry_receives_a_checkpoints_dir(self, tmp_path) -> None:
        """Defect 2: `FilesystemModelRegistry()` is missing its required
        `checkpoints_dir` argument; the default must resolve one (explicit
        override here) instead of crashing with TypeError.
        """
        from HWFP.cli.bot_main import build_container
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        container = build_container(
            token="fake-token",
            chat_id=1,
            checkpoints_dir=tmp_path,
            partidos_source=_stub_partidos,
            states=object(),
            tracker=object(),
            bot=object(),
            scheduler=object(),
        )
        assert isinstance(container.model_registry, FilesystemModelRegistry)
        assert container.model_registry._checkpoints_dir == tmp_path

    def test_default_model_registry_falls_back_to_default_checkpoints_dir(
        self,
    ) -> None:
        """When no explicit checkpoints_dir is given, build_container() must
        resolve HWFP.models.paths.default_checkpoints_dir() — never crash
        with a missing-argument TypeError as the old `FilesystemModelRegistry()`
        call did.
        """
        from HWFP.cli.bot_main import build_container
        from HWFP.models.paths import default_checkpoints_dir
        from HWFP.serving.adapters.filesystem_model_registry import (
            FilesystemModelRegistry,
        )

        container = build_container(
            token="fake-token",
            chat_id=1,
            partidos_source=_stub_partidos,
            states=object(),
            tracker=object(),
            bot=object(),
            scheduler=object(),
        )
        assert isinstance(container.model_registry, FilesystemModelRegistry)
        assert container.model_registry._checkpoints_dir == default_checkpoints_dir()


class TestResolveCheckpointsDir:
    """Pure-function unit tests for the HWFP_CHECKPOINTS_DIR env override
    (task 4.2), extracted so it is testable without invoking main() (which
    blocks on bot.run_polling() and requires real Telegram env vars).
    """

    def test_returns_env_override_when_set(self, monkeypatch, tmp_path) -> None:
        from HWFP.cli.bot_main import _resolve_checkpoints_dir

        monkeypatch.setenv("HWFP_CHECKPOINTS_DIR", str(tmp_path))
        assert _resolve_checkpoints_dir() == tmp_path

    def test_returns_default_checkpoints_dir_when_unset(self, monkeypatch) -> None:
        from HWFP.cli.bot_main import _resolve_checkpoints_dir
        from HWFP.models.paths import default_checkpoints_dir

        monkeypatch.delenv("HWFP_CHECKPOINTS_DIR", raising=False)
        assert _resolve_checkpoints_dir() == default_checkpoints_dir()
