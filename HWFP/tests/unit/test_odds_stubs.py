"""Tests for odds stubs and AggregatedOddsProvider."""

from __future__ import annotations

from datetime import datetime

import pytest

from HWFP.core.domain.odds import Odds
from HWFP.serving.adapters.aggregated_odds_provider import AggregatedOddsProvider
from HWFP.serving.adapters.cgm_odds_stub import CGMOddsStub
from HWFP.serving.adapters.codere_odds_stub import CodereOddsStub
from HWFP.serving.adapters.onexbet_odds_stub import OnexBetOddsStub


def _make_odds(decimal: float, bookmaker: str) -> Odds:
    return Odds(
        match_id="m1",
        market="total_fouls",
        line=2.5,
        side="over",
        decimal=decimal,
        bookmaker=bookmaker,
        fetched_at=datetime(2026, 1, 1),
    )


# ── Individual stub smoke tests ────────────────────────────────────────────────


class TestCodereOddsStub:
    def test_returns_odds_object(self) -> None:
        odds = CodereOddsStub().get_odds("m1", "total_fouls")
        assert isinstance(odds, Odds)

    def test_bookmaker_is_codere(self) -> None:
        odds = CodereOddsStub().get_odds("m1", "total_fouls")
        assert odds.bookmaker == "codere"

    def test_decimal_above_one(self) -> None:
        odds = CodereOddsStub().get_odds("m1", "total_fouls")
        assert odds.decimal > 1.0

    def test_match_id_propagated(self) -> None:
        odds = CodereOddsStub().get_odds("match-99", "total_fouls")
        assert odds.match_id == "match-99"


class TestCGMOddsStub:
    def test_returns_odds_object(self) -> None:
        odds = CGMOddsStub().get_odds("m1", "total_fouls")
        assert isinstance(odds, Odds)

    def test_bookmaker_is_cgm(self) -> None:
        odds = CGMOddsStub().get_odds("m1", "total_fouls")
        assert odds.bookmaker == "cgm"

    def test_decimal_above_one(self) -> None:
        odds = CGMOddsStub().get_odds("m1", "total_fouls")
        assert odds.decimal > 1.0


class TestOnexBetOddsStub:
    def test_returns_odds_object(self) -> None:
        odds = OnexBetOddsStub().get_odds("m1", "total_fouls")
        assert isinstance(odds, Odds)

    def test_bookmaker_is_1xbet(self) -> None:
        odds = OnexBetOddsStub().get_odds("m1", "total_fouls")
        assert odds.bookmaker == "1xbet"

    def test_decimal_above_one(self) -> None:
        odds = OnexBetOddsStub().get_odds("m1", "total_fouls")
        assert odds.decimal > 1.0


# ── AggregatedOddsProvider ─────────────────────────────────────────────────────


class TestAggregatedOddsProvider:
    def test_all_three_stubs_return_three_odds(self) -> None:
        aggregator = AggregatedOddsProvider(
            [CodereOddsStub(), CGMOddsStub(), OnexBetOddsStub()]
        )
        results = aggregator.get_odds("m1", "total_fouls")
        assert len(results) == 3

    def test_best_odds_returns_highest_decimal_among_stubs(self) -> None:
        # 1xbet mock decimal (1.92) > Codere (1.90) > CGM (1.87)
        aggregator = AggregatedOddsProvider(
            [CodereOddsStub(), CGMOddsStub(), OnexBetOddsStub()]
        )
        best = aggregator.best_odds("m1", "total_fouls")
        assert best is not None
        assert best.bookmaker == "1xbet"

    def test_real_provider_beats_mock_stubs_when_better(self) -> None:
        class _BetterProvider:
            def get_odds(self, match_id: str, market: str) -> Odds:
                return _make_odds(2.20, "premium_book")

        aggregator = AggregatedOddsProvider(
            [CodereOddsStub(), CGMOddsStub(), _BetterProvider()]
        )
        best = aggregator.best_odds("m1", "total_fouls")
        assert best is not None
        assert best.bookmaker == "premium_book"
        assert best.decimal == pytest.approx(2.20)

    def test_best_odds_returns_highest_decimal(self) -> None:
        low = _make_odds(1.75, "bookie_a")
        high = _make_odds(2.10, "bookie_b")

        class _ProviderA:
            def get_odds(self, match_id: str, market: str) -> Odds:
                return low

        class _ProviderB:
            def get_odds(self, match_id: str, market: str) -> Odds:
                return high

        aggregator = AggregatedOddsProvider([_ProviderA(), _ProviderB()])
        assert aggregator.best_odds("m1", "total_fouls") == high

    def test_empty_providers_returns_none(self) -> None:
        aggregator = AggregatedOddsProvider([])
        assert aggregator.best_odds("m1", "total_fouls") is None

    def test_error_raising_provider_skipped(self) -> None:
        class _BrokenProvider:
            def get_odds(self, match_id: str, market: str) -> Odds:
                raise NotImplementedError("broken")

        aggregator = AggregatedOddsProvider([_BrokenProvider(), CodereOddsStub()])
        results = aggregator.get_odds("m1", "total_fouls")
        assert len(results) == 1
        assert results[0].bookmaker == "codere"
