"""ListMatchProvider — static list-backed MatchProvider for dev and Railway config."""

from __future__ import annotations

import os
from datetime import datetime

from HWFP.core.ports.match_provider import MatchConfig, MatchProvider


class ListMatchProvider:
    """MatchProvider backed by an explicit list of configs.

    In production, populate from env var HWFP_MATCH_IDS (comma-separated match IDs)
    or pass configs directly at construction time.

    Example env:
        HWFP_MATCH_IDS=laliga-2025-38-001,laliga-2025-38-002
        HWFP_MATCH_MARKET=total_fouls
        HWFP_MATCH_SIDE=over
    """

    def __init__(self, configs: list[MatchConfig] | None = None) -> None:
        if configs is not None:
            self._configs = configs
        else:
            self._configs = _configs_from_env()

    def get_upcoming_configs(self, date: datetime) -> list[MatchConfig]:
        return list(self._configs)


def _configs_from_env() -> list[MatchConfig]:
    raw = os.environ.get("HWFP_MATCH_IDS", "")
    market = os.environ.get("HWFP_MATCH_MARKET", "total_fouls")
    side = os.environ.get("HWFP_MATCH_SIDE", "over")
    if not raw.strip():
        return []
    return [
        MatchConfig(match_id=mid.strip(), market=market, side=side)
        for mid in raw.split(",")
        if mid.strip()
    ]
