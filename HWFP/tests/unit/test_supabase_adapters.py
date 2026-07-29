"""Unit and integration tests for Supabase adapters.

TDD RED phase — tests are written before the implementations.
Integration tests are skipped when SUPABASE_URL is not set.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pytest

from HWFP.core.domain.bet_record import BetOutcome
from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.confidence_score import ConfidenceLevel, ConfidenceScore
from HWFP.core.domain.exceptions import StateNotFoundError
from HWFP.serving.adapters.supabase_performance_tracker import SupabasePerformanceTracker
from HWFP.serving.adapters.supabase_prediction_sink import SupabasePredictionSink
from HWFP.serving.adapters.supabase_state_adapter import SupabaseStateAdapter

supabase_available = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL"),
    reason="SUPABASE_URL not set",
)

# ---------------------------------------------------------------------------
# Fake Supabase infrastructure
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class _FakeQuery:
    """Fluent Supabase/postgrest-py query builder backed by in-memory data."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        sink: dict[str, list[dict[str, Any]]],
        table: str,
    ) -> None:
        self._rows = rows
        self._sink = sink
        self._table = table
        self._filters: list[tuple[str, str, Any]] = []
        self._mode = "select"
        self._write_data: list[dict[str, Any]] = []
        self._limit_n: int | None = None

    def select(self, *args: Any, **kwargs: Any) -> _FakeQuery:
        return self

    def eq(self, col: str, val: Any) -> _FakeQuery:
        self._filters.append(("eq", col, val))
        return self

    def lt(self, col: str, val: Any) -> _FakeQuery:
        self._filters.append(("lt", col, val))
        return self

    def or_(self, clause: str, **kwargs: Any) -> _FakeQuery:
        return self

    def order(self, *args: Any, **kwargs: Any) -> _FakeQuery:
        return self

    def limit(self, n: int) -> _FakeQuery:
        self._limit_n = n
        return self

    def upsert(self, data: list[dict[str, Any]] | dict[str, Any], **kwargs: Any) -> _FakeQuery:
        self._mode = "write"
        self._write_data = data if isinstance(data, list) else [data]
        return self

    def insert(self, data: list[dict[str, Any]] | dict[str, Any], **kwargs: Any) -> _FakeQuery:
        self._mode = "write"
        self._write_data = data if isinstance(data, list) else [data]
        return self

    def update(self, data: dict[str, Any], **kwargs: Any) -> _FakeQuery:
        self._mode = "write"
        self._write_data = [data]
        return self

    def execute(self) -> _FakeResponse:
        if self._mode == "write":
            self._sink.setdefault(self._table, []).extend(self._write_data)
            return _FakeResponse(self._write_data)
        rows = list(self._rows)
        for op, col, val in self._filters:
            if op == "eq":
                rows = [r for r in rows if r.get(col) == val]
            elif op == "lt":
                rows = [r for r in rows if (r.get(col) or "") < str(val)]
        if self._limit_n is not None:
            rows = rows[: self._limit_n]
        return _FakeResponse(rows)


class FakeSupabaseClient:
    """In-memory Supabase client for unit tests. Tracks all writes in `.writes`."""

    def __init__(self, tables: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = tables or {}
        self.writes: dict[str, list[dict[str, Any]]] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(list(self._tables.get(name, [])), self.writes, name)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_decision() -> BettingDecision:
    return BettingDecision(
        match_id="m1",
        recommendation=Recommendation.BET,
        market="fouls",
        line=25.5,
        side="over",
        best_bookmaker="codere",
        best_odds_value=1.85,
        p_model=0.58,
        edge=0.07,
        stake_euros=10.0,
        confidence=ConfidenceScore(
            pmf_entropy=0.3,
            referee_sample_size=20,
            feature_fallback_count=0,
            kelly_multiplier=0.25,
            level=ConfidenceLevel.HIGH,
        ),
        reasons=("edge > threshold",),
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _make_row(bet_id: str, outcome: str = "pending") -> dict[str, Any]:
    return {
        "bet_id": bet_id,
        "match_id": "m1",
        "recommendation": "bet",
        "market": "fouls",
        "line": 25.5,
        "side": "over",
        "best_bookmaker": "codere",
        "best_odds_value": 1.85,
        "p_model": 0.58,
        "edge": 0.07,
        "stake_euros": 10.0,
        "confidence_level": "high",
        "kelly_multiplier": 0.25,
        "pmf_entropy": 0.3,
        "referee_sample_size": 20,
        "feature_fallback_count": 0,
        "reasons_json": "[]",
        "generated_at": "2024-01-01T00:00:00",
        "placed_at": "2024-01-01T00:00:00",
        "outcome": outcome,
        "closing_line": None,
        "closing_odds": None,
        "clv": None,
        "profit_euros": None,
    }


# ---------------------------------------------------------------------------
# SupabaseStateAdapter — unit tests
# ---------------------------------------------------------------------------


class TestSupabaseStateAdapterInit:
    def test_constructor_does_not_connect(self) -> None:
        adapter = SupabaseStateAdapter("http://fake", "fake-key", _client=FakeSupabaseClient())
        assert adapter is not None

    def test_constructor_stores_timeout(self) -> None:
        adapter = SupabaseStateAdapter(
            "http://fake", "fake-key", timeout_seconds=5.0, _client=FakeSupabaseClient()
        )
        assert adapter._timeout == 5.0


class TestSupabaseStateAdapterGetMatch:
    def test_raises_state_not_found_when_no_rows(self) -> None:
        client = FakeSupabaseClient(tables={"matches": []})
        adapter = SupabaseStateAdapter("http://fake", "fake-key", _client=client)
        with pytest.raises(StateNotFoundError):
            adapter.get_match("nonexistent-id")

    def test_returns_match_domain_object(self) -> None:
        row = {
            "match_id": "m1",
            "home_team": "TeamA",
            "away_team": "TeamB",
            "match_date": "2024-01-15T15:00:00",
            "referee": "ref1",
            "season": "2023-24",
            "fouls_home": 10,
            "fouls_away": 8,
        }
        client = FakeSupabaseClient(tables={"matches": [row]})
        adapter = SupabaseStateAdapter("http://fake", "fake-key", _client=client)
        match = adapter.get_match("m1")
        assert match.match_id == "m1"
        assert match.home_team_id == "TeamA"
        assert match.away_team_id == "TeamB"

    def test_missing_referee_uses_unknown_fallback(self) -> None:
        row = {
            "match_id": "m1",
            "home_team": "TeamA",
            "away_team": "TeamB",
            "match_date": "2024-01-15T15:00:00",
            "referee": None,
            "season": "2023-24",
        }
        client = FakeSupabaseClient(tables={"matches": [row]})
        adapter = SupabaseStateAdapter("http://fake", "fake-key", _client=client)
        match = adapter.get_match("m1")
        assert match.referee_id == "unknown"


class TestSupabaseStateAdapterGetTeamState:
    def _make_match_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "match_id": f"m{i}",
                "home_team": "TeamA",
                "away_team": "TeamB",
                "match_date": f"2024-01-0{i + 1}T15:00:00",
                "referee": "ref1",
                "season": "2023-24",
                "fouls_home": 10,
                "fouls_away": 8,
            }
            for i in range(3)
        ]

    def test_returns_team_state_with_non_negative_averages(self) -> None:
        client = FakeSupabaseClient(tables={"matches": self._make_match_rows()})
        adapter = SupabaseStateAdapter("http://fake", "fake-key", _client=client)
        state = adapter.get_team_state("TeamA", datetime(2024, 2, 1, tzinfo=timezone.utc))
        assert state.team_id == "TeamA"
        assert state.avg_fouls_per_match >= 0.0
        assert state.avg_fouls_conceded >= 0.0

    def test_averages_computed_from_home_matches(self) -> None:
        client = FakeSupabaseClient(tables={"matches": self._make_match_rows()})
        adapter = SupabaseStateAdapter("http://fake", "fake-key", _client=client)
        state = adapter.get_team_state("TeamA", datetime(2024, 2, 1, tzinfo=timezone.utc))
        assert state.avg_fouls_per_match == pytest.approx(10.0)
        assert state.avg_fouls_conceded == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# SupabasePredictionSink — unit tests
# ---------------------------------------------------------------------------


def _make_prediction() -> Any:
    from HWFP.core.domain.foul_pmf import FoulPMF
    from HWFP.core.domain.match_prediction import MatchPrediction
    from HWFP.core.domain.model_id import ModelId

    pmf = FoulPMF(pmf=(0.5, 0.5), bin_edges=(0, 5, 10))
    return MatchPrediction(
        match_id="m1",
        pmf=pmf,
        model_id=ModelId("test-model"),
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class TestSupabasePredictionSink:
    def test_write_targets_predictions_table(self) -> None:
        client = FakeSupabaseClient()
        sink = SupabasePredictionSink("http://fake", "fake-key", _client=client)
        sink.write(_make_prediction())
        assert "predictions" in client.writes

    def test_write_does_not_raise_on_missing_table(self) -> None:
        client = FakeSupabaseClient()
        sink = SupabasePredictionSink("http://fake", "fake-key", _client=client)
        sink.write(_make_prediction())  # must not raise

    def test_write_includes_match_id(self) -> None:
        client = FakeSupabaseClient()
        sink = SupabasePredictionSink("http://fake", "fake-key", _client=client)
        sink.write(_make_prediction())
        assert client.writes["predictions"][0]["match_id"] == "m1"


# ---------------------------------------------------------------------------
# SupabasePerformanceTracker — unit tests
# ---------------------------------------------------------------------------


class TestSupabasePerformanceTracker:
    def test_record_bet_returns_pending_outcome(self) -> None:
        client = FakeSupabaseClient()
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        record = tracker.record_bet(_make_decision())
        assert record.outcome == BetOutcome.PENDING

    def test_record_bet_writes_to_bet_records_table(self) -> None:
        client = FakeSupabaseClient()
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        tracker.record_bet(_make_decision())
        assert "bet_records" in client.writes

    def test_record_bet_returns_same_decision(self) -> None:
        client = FakeSupabaseClient()
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        decision = _make_decision()
        record = tracker.record_bet(decision)
        assert record.decision == decision

    def test_record_bet_generates_unique_bet_ids(self) -> None:
        client = FakeSupabaseClient()
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        r1 = tracker.record_bet(_make_decision())
        r2 = tracker.record_bet(_make_decision())
        assert r1.bet_id != r2.bet_id

    def test_get_pending_records_returns_only_pending(self) -> None:
        rows = [_make_row("b1", "pending"), _make_row("b2", "win")]
        client = FakeSupabaseClient(tables={"bet_records": rows})
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        pending = tracker.get_pending_records()
        assert len(pending) == 1
        assert pending[0].outcome == BetOutcome.PENDING

    def test_get_records_returns_all(self) -> None:
        rows = [_make_row(f"b{i}") for i in range(5)]
        client = FakeSupabaseClient(tables={"bet_records": rows})
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        assert len(tracker.get_records()) == 5

    def test_get_records_last_n_limits_results(self) -> None:
        rows = [_make_row(f"b{i}") for i in range(5)]
        client = FakeSupabaseClient(tables={"bet_records": rows})
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        assert len(tracker.get_records(last_n=3)) == 3

    def test_update_outcome_writes_to_sink(self) -> None:
        client = FakeSupabaseClient(tables={"bet_records": [_make_row("b1")]})
        tracker = SupabasePerformanceTracker("http://fake", "fake-key", _client=client)
        tracker.update_outcome("b1", BetOutcome.WIN)
        assert "bet_records" in client.writes


# ---------------------------------------------------------------------------
# Integration tests (require SUPABASE_URL env var)
# ---------------------------------------------------------------------------


@supabase_available
class TestSupabaseStateAdapterIntegration:
    def test_get_match_raises_not_found_for_unknown_id(self) -> None:
        adapter = SupabaseStateAdapter(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
        with pytest.raises(StateNotFoundError):
            adapter.get_match("__nonexistent_integration_test_id__")


@supabase_available
class TestSupabasePredictionSinkIntegration:
    def test_write_does_not_crash(self) -> None:
        sink = SupabasePredictionSink(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
        sink.write(_make_prediction())  # table may not exist — must not raise


@supabase_available
class TestSupabasePerformanceTrackerIntegration:
    def test_record_and_retrieve_pending(self) -> None:
        tracker = SupabasePerformanceTracker(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
        record = tracker.record_bet(_make_decision())
        assert record.outcome == BetOutcome.PENDING
        pending = tracker.get_pending_records()
        assert any(r.bet_id == record.bet_id for r in pending)
