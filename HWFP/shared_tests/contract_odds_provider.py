"""Contract tests for OddsProvider port (REQ-8, REQ-9)."""

from __future__ import annotations

import pytest

from HWFP.core.domain.exceptions import OddsNotFoundError
from HWFP.core.domain.odds import Odds

_MATCH_ID = "M1"
_MARKET = "fouls_over_under"


@pytest.fixture(params=["fake", "stub"], ids=["fake", "stub"])
def odds_provider(request):
    if request.param == "fake":
        mod = pytest.importorskip("HWFP.serving.fakes.fake_odds_provider")
        return mod.FakeOddsProvider.with_fixture()
    pytest.importorskip("HWFP.serving.adapters.codere_odds_adapter")
    pytest.skip("stub_adapter: raises NotImplementedError by design")


def test_get_odds_returns_odds(odds_provider):
    result = odds_provider.get_odds(_MATCH_ID, _MARKET)
    assert isinstance(result, Odds)


def test_get_odds_decimal_gt_one(odds_provider):
    result = odds_provider.get_odds(_MATCH_ID, _MARKET)
    assert result.decimal > 1.0


def test_get_odds_unknown_match_raises(odds_provider):
    with pytest.raises(OddsNotFoundError):
        odds_provider.get_odds("unknown-match-xyz-99", _MARKET)


def test_get_odds_unknown_market_raises(odds_provider):
    with pytest.raises(OddsNotFoundError):
        odds_provider.get_odds(_MATCH_ID, "unknown-market-xyz-99")


def test_odds_provider_stub_raises_not_implemented():
    mod = pytest.importorskip("HWFP.serving.adapters.codere_odds_adapter")
    adapter = mod.CodereOddsAdapter()
    with pytest.raises(NotImplementedError):
        adapter.get_odds("any-match-id", "any-market")
