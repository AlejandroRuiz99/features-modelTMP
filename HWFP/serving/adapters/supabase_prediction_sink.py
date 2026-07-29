"""Supabase adapter — PredictionSink port.

Upserts MatchPrediction records to the Supabase `predictions` table.
If the table does not exist yet, the error is logged as a warning instead
of propagating, so the calling pipeline is not disrupted.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from typing import Any

from HWFP.core.domain.match_prediction import MatchPrediction

logger = logging.getLogger(__name__)

_TABLE = "predictions"


class SupabasePredictionSink:
    """PredictionSink backed by the Supabase `predictions` table.

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
                max_workers=2, thread_name_prefix="supa-sink"
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

    def write(self, prediction: MatchPrediction) -> None:
        """Upsert a MatchPrediction to Supabase.

        Logs a warning instead of raising if the `predictions` table does not
        exist or any other transient error occurs.

        Args:
            prediction: The prediction to persist.
        """
        row: dict[str, Any] = {
            "match_id": prediction.match_id,
            "model_id": prediction.model_id.value,
            "generated_at": prediction.generated_at.isoformat(),
            "pmf_values": list(prediction.pmf.pmf),
            "bin_edges": list(prediction.pmf.bin_edges),
        }
        sb = self._get_client()

        def _query() -> Any:
            return sb.table(_TABLE).upsert(row).execute()

        try:
            self._run(_query)
        except Exception as exc:
            logger.warning(
                "Failed to write prediction for match %s to Supabase "
                "(table '%s' may not exist yet): %s",
                prediction.match_id,
                _TABLE,
                exc,
            )
