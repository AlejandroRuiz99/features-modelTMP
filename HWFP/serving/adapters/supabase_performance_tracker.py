"""Supabase adapter — PerformanceTracker port.

Stores and retrieves BetRecord rows in the Supabase `bet_records` table.

Table schema (create via migration or Supabase Studio):
    bet_id              text PRIMARY KEY,
    match_id            text NOT NULL,
    recommendation      text NOT NULL,
    market              text NOT NULL,
    line                numeric NOT NULL,
    side                text NOT NULL,
    best_bookmaker      text NOT NULL,
    best_odds_value     numeric NOT NULL,
    p_model             numeric NOT NULL,
    edge                numeric NOT NULL,
    stake_euros         numeric NOT NULL,
    confidence_level    text NOT NULL,
    kelly_multiplier    numeric NOT NULL,
    pmf_entropy         numeric NOT NULL,
    referee_sample_size integer NOT NULL,
    feature_fallback_count integer NOT NULL,
    reasons_json        text NOT NULL,
    generated_at        timestamptz NOT NULL,
    placed_at           timestamptz NOT NULL,
    outcome             text NOT NULL DEFAULT 'pending',
    closing_line        numeric,
    closing_odds        numeric,
    clv                 numeric,
    profit_euros        numeric
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from HWFP.core.domain.bet_record import BetOutcome, BetRecord
from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.confidence_score import ConfidenceLevel, ConfidenceScore

logger = logging.getLogger(__name__)

_TABLE = "bet_records"


class SupabasePerformanceTracker:
    """PerformanceTracker backed by the Supabase `bet_records` table.

    Args:
        url: Supabase project URL (SUPABASE_URL).
        key: Supabase service-role or anon key (SUPABASE_KEY).
        timeout_seconds: Per-call network timeout in seconds.
        _client: Injected client for testing; skips lazy init and executor.
    """

    def __init__(
        self,
        url: str,
        key: str,
        timeout_seconds: float = 10.0,
        *,
        _client: object | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._timeout = timeout_seconds
        self._injected: object | None = _client
        self._client: Any = _client
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            None
            if _client is not None
            else concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="supa-tracker"
            )
        )

    def _get_client(self) -> Any:
        if self._client is None:
            from supabase import create_client

            self._client = create_client(self._url, self._key)
        return self._client

    def _run(self, fn: Callable[[], Any]) -> Any:
        if self._executor is None:
            return fn()
        future = self._executor.submit(fn)
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"Supabase call timed out after {self._timeout}s"
            ) from exc

    def record_bet(self, decision: BettingDecision) -> BetRecord:
        """Persist a new bet decision with PENDING outcome.

        Args:
            decision: The BettingDecision to record.

        Returns:
            BetRecord with a generated bet_id and outcome=PENDING.
        """
        bet_id = str(uuid.uuid4())
        placed_at = datetime.now(tz=timezone.utc)
        row = _decision_to_row(bet_id, decision, placed_at)
        sb = self._get_client()

        def _query() -> Any:
            return sb.table(_TABLE).insert(row).execute()

        self._run(_query)
        return BetRecord(
            bet_id=bet_id,
            decision=decision,
            placed_at=placed_at,
            outcome=BetOutcome.PENDING,
            closing_line=None,
            closing_odds=None,
            clv=None,
            profit_euros=None,
        )

    def update_outcome(self, bet_id: str, outcome: BetOutcome) -> None:
        """Update the outcome field of an existing bet record.

        Args:
            bet_id: The bet_id of the record to update.
            outcome: The resolved BetOutcome value.
        """
        sb = self._get_client()

        def _query() -> Any:
            return (
                sb.table(_TABLE)
                .update({"outcome": outcome.value})
                .eq("bet_id", bet_id)
                .execute()
            )

        self._run(_query)

    def get_records(self, last_n: int | None = None) -> list[BetRecord]:
        """Retrieve bet records ordered by placed_at descending.

        Args:
            last_n: If provided, return only the most recent N records.

        Returns:
            List of BetRecord domain objects.
        """
        sb = self._get_client()

        def _query() -> Any:
            q = sb.table(_TABLE).select("*").order("placed_at", desc=True)
            if last_n is not None:
                q = q.limit(last_n)
            return q.execute()

        response = self._run(_query)
        return [_row_to_bet_record(r) for r in (response.data or [])]

    def get_pending_records(self) -> list[BetRecord]:
        """Retrieve all records with outcome=PENDING.

        Returns:
            List of BetRecord domain objects whose outcome is PENDING.
        """
        sb = self._get_client()

        def _query() -> Any:
            return (
                sb.table(_TABLE)
                .select("*")
                .eq("outcome", BetOutcome.PENDING.value)
                .execute()
            )

        response = self._run(_query)
        return [_row_to_bet_record(r) for r in (response.data or [])]


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------


def _decision_to_row(
    bet_id: str,
    decision: BettingDecision,
    placed_at: datetime,
) -> dict[str, Any]:
    return {
        "bet_id": bet_id,
        "match_id": decision.match_id,
        "recommendation": decision.recommendation.value,
        "market": decision.market,
        "line": decision.line,
        "side": decision.side,
        "best_bookmaker": decision.best_bookmaker,
        "best_odds_value": decision.best_odds_value,
        "p_model": decision.p_model,
        "edge": decision.edge,
        "stake_euros": decision.stake_euros,
        "confidence_level": decision.confidence.level.value,
        "kelly_multiplier": decision.confidence.kelly_multiplier,
        "pmf_entropy": decision.confidence.pmf_entropy,
        "referee_sample_size": decision.confidence.referee_sample_size,
        "feature_fallback_count": decision.confidence.feature_fallback_count,
        "reasons_json": json.dumps(list(decision.reasons)),
        "generated_at": decision.generated_at.isoformat(),
        "placed_at": placed_at.isoformat(),
        "outcome": BetOutcome.PENDING.value,
        "closing_line": None,
        "closing_odds": None,
        "clv": None,
        "profit_euros": None,
    }


def _row_to_bet_record(row: dict[str, Any]) -> BetRecord:
    """Reconstruct a BetRecord domain object from a Supabase row."""
    placed_at = _parse_dt(row.get("placed_at") or row.get("generated_at") or "")
    generated_at = _parse_dt(row.get("generated_at") or "")

    reasons_raw = row.get("reasons_json") or "[]"
    try:
        reasons: tuple[str, ...] = tuple(json.loads(reasons_raw))
    except (json.JSONDecodeError, TypeError):
        reasons = ()

    decision = BettingDecision(
        match_id=row.get("match_id") or "",
        recommendation=Recommendation(row.get("recommendation") or Recommendation.SKIP.value),
        market=row.get("market") or "",
        line=float(row.get("line") or 0.0),
        side=row.get("side") or "",
        best_bookmaker=row.get("best_bookmaker") or "",
        best_odds_value=float(row.get("best_odds_value") or 0.0),
        p_model=float(row.get("p_model") or 0.0),
        edge=float(row.get("edge") or 0.0),
        stake_euros=float(row.get("stake_euros") or 0.0),
        confidence=ConfidenceScore(
            pmf_entropy=float(row.get("pmf_entropy") or 0.0),
            referee_sample_size=int(row.get("referee_sample_size") or 0),
            feature_fallback_count=int(row.get("feature_fallback_count") or 0),
            kelly_multiplier=float(row.get("kelly_multiplier") or 0.0),
            level=ConfidenceLevel(row.get("confidence_level") or ConfidenceLevel.LOW.value),
        ),
        reasons=reasons,
        generated_at=generated_at,
    )
    return BetRecord(
        bet_id=row.get("bet_id") or "",
        decision=decision,
        placed_at=placed_at,
        outcome=BetOutcome(row.get("outcome") or BetOutcome.PENDING.value),
        closing_line=row.get("closing_line"),
        closing_odds=row.get("closing_odds"),
        clv=row.get("clv"),
        profit_euros=row.get("profit_euros"),
    )


def _parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
