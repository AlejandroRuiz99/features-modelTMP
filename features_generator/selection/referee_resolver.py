"""Resolucion de arbitros via API externa RFEF."""

from __future__ import annotations

import logging
from typing import Any

import requests

from core.utils import team_match

_REFEREE_API_URL = "https://api-referees.vercel.app/"
logger = logging.getLogger(__name__)


def _fetch_referees_for_date(date_iso: str | None = None) -> list[dict[str, Any]]:
    params = {"date": date_iso} if date_iso else None
    try:
        r = requests.get(_REFEREE_API_URL, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
    except Exception:
        logger.warning(
            "Failed to fetch referees from API for date %s", date_iso, exc_info=True
        )
        return []
    if not isinstance(j, dict):
        return []
    matches = j.get("matches", [])
    if not isinstance(matches, list):
        return []
    out = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        out.append(
            {
                "date": m.get("date"),
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "referee": m.get("referee"),
            }
        )
    return out


def resolve_referee_from_external_api(
    equipo_local: str,
    equipo_visitante: str,
    date_iso: str | None = None,
) -> str | None:
    """Busca el arbitro designado para un partido via API RFEF."""
    daily = _fetch_referees_for_date(date_iso)
    if not daily:
        return None
    for m in daily:
        home = m.get("home_team")
        away = m.get("away_team")
        same_pair = (
            team_match(home, equipo_local) and team_match(away, equipo_visitante)
        ) or (team_match(home, equipo_visitante) and team_match(away, equipo_local))
        if same_pair and m.get("referee"):
            return str(m["referee"]).strip()
    return None
