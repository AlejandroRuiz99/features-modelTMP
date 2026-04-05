"""Fetch de cuotas de apuestas desde Supabase."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from selection import supabase_client
from transformation import buscar_equipo
from core.utils import team_match


def get_match_odds_rows(
    scores: dict[str, Any],
    equipo_local: str,
    equipo_visitante: str,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Busca las filas de cuotas para un partido concreto en Supabase.

    Returns:
        (rows_filtradas, scraped_at, event_id)
    """
    odds = supabase_client.fetch_latest_odds_rows(sport="soccer")
    rows = odds.get("rows", [])
    scraped_at = odds.get("scraped_at")
    if not rows:
        return [], scraped_at, None

    def resolve(n: str | None) -> str:
        return buscar_equipo(n or "", scores) or (n or "")

    def _filter(local: str, visitante: str) -> list[dict[str, Any]]:
        return [
            r
            for r in rows
            if team_match(resolve(r.get("home_team")), local)
            and team_match(resolve(r.get("away_team")), visitante)
        ]

    match_rows = _filter(equipo_local, equipo_visitante)
    if not match_rows:
        # El bookmaker a veces invierte local/visitante — intentar al revés
        match_rows = _filter(equipo_visitante, equipo_local)
    if not match_rows:
        return [], scraped_at, None

    by_event: dict[str, int] = defaultdict(int)
    for r in match_rows:
        ev = (r.get("external_event_id") or "").strip()
        if ev:
            by_event[ev] += 1
    event_id = None
    if by_event:
        event_id = max(by_event.items(), key=lambda x: x[1])[0]
        match_rows = [
            r
            for r in match_rows
            if (r.get("external_event_id") or "").strip() == event_id
        ]
    return match_rows, scraped_at, event_id
