"""Unit tests for HWFP.features.assembly.betting_odds — market data source DI.

Spec capability `architecture-boundaries` (MODIFIED), REQ-14 awareness item
(session-scoped fix, tracked since Batch 2/4/5): `build_market_category()`
used to reach a call-time `from selection.odds_client import
get_match_odds_rows`, which is genuinely broken outside of pytest's
`pythonpath` config (no top-level `selection` package exists at runtime) —
every real (non-`skip_market_fetch=True`) feature build crashed with
`ModuleNotFoundError`.

TDD cycle:
  RED   — `set_market_data_source` did not exist; `build_market_category()`
          always attempted the broken legacy import regardless of whether a
          real market source was configured.
  GREEN — `HWFP.features.assembly.betting_odds` now exposes an injectable
          data source (mirroring `HWFP.features.core.state_cache`'s
          `set_data_source` pattern from Batch 2/D1): the composition root
          wires a real fetcher; without one, `build_market_category()`
          raises a clear `RuntimeError` instead of a cryptic
          `ModuleNotFoundError` reaching into a package that was never
          importable in production.
"""

from __future__ import annotations

import pytest

from HWFP.features.assembly import betting_odds


@pytest.fixture(autouse=True)
def _reset_market_data_source():
    """Isolate the module-level injected fetch fn between tests."""
    betting_odds._market_fetch_fn = None
    yield
    betting_odds._market_fetch_fn = None


class TestNoDataSourceConfigured:
    def test_raises_clear_error_when_no_market_data_source_configured(self) -> None:
        with pytest.raises(RuntimeError, match="market data source"):
            betting_odds.build_market_category(
                scores={},
                eq_local="Real Madrid",
                eq_visit="Barcelona",
                model_market_signal={},
            )


class TestSetMarketDataSource:
    def test_injected_fetch_fn_backs_build_market_category(self) -> None:
        """Scenario: with a source injected, build_market_category() uses
        its rows instead of raising — proves the DI hook is load-bearing,
        not a no-op stub.
        """

        def _stub_fetch(scores, eq_local, eq_visit, *, match_date=None):
            rows = [
                {"mercado": "1X2", "selection": "local", "cuota": "2.10"},
                {"mercado": "1X2", "selection": "empate", "cuota": "3.30"},
                {"mercado": "1X2", "selection": "visitante", "cuota": "3.40"},
            ]
            return rows, "2025-01-01T00:00:00", "evt-1"

        betting_odds.set_market_data_source(_stub_fetch)

        market, market_input_model, used, scraped_at = betting_odds.build_market_category(
            scores={},
            eq_local="Real Madrid",
            eq_visit="Barcelona",
            model_market_signal={},
        )

        assert "1x2" in used
        assert scraped_at == "2025-01-01T00:00:00"
        assert market["traza"]["event_id"] == "evt-1"

    def test_injected_fetch_fn_receives_scores_and_teams(self) -> None:
        """Triangulation: a second scenario with different inputs proves the
        DI hook actually forwards its arguments, not just returns canned
        output regardless of call site.
        """
        captured: dict = {}

        def _capturing_fetch(scores, eq_local, eq_visit, *, match_date=None):
            captured["scores"] = scores
            captured["eq_local"] = eq_local
            captured["eq_visit"] = eq_visit
            captured["match_date"] = match_date
            return [], None, None

        betting_odds.set_market_data_source(_capturing_fetch)

        betting_odds.build_market_category(
            scores={"marker": True},
            eq_local="Sevilla",
            eq_visit="Valencia",
            model_market_signal={},
            match_date="2025-03-15",
        )

        assert captured == {
            "scores": {"marker": True},
            "eq_local": "Sevilla",
            "eq_visit": "Valencia",
            "match_date": "2025-03-15",
        }

    def test_empty_rows_from_data_source_yield_no_used_markets(self) -> None:
        """Edge case: a real source that returns no rows for this match
        must not crash and must not report any markets as usable.
        """
        betting_odds.set_market_data_source(lambda *a, **k: ([], None, None))

        _market, _model, used, scraped_at = betting_odds.build_market_category(
            scores={},
            eq_local="Real Madrid",
            eq_visit="Barcelona",
            model_market_signal={},
        )

        assert used == []
        assert scraped_at is None
