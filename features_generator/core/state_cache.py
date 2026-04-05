"""Estado estadistico precalculado (singleton thread-safe).

build_state()  — computa scores, rankings, xstyles, arbitros, GMM, calendario
                 y objectives a partir de una lista de partidos (puro, sin I/O).
get_state()    — devuelve el estado cacheado; carga de Supabase la primera vez.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

from transformation import calcular_scores, calcular_xstyle
from transformation.referees import calcular_perfiles as calcular_perfiles_arbitros
from transformation.referee_gmm import calcular_perfiles_gmm
from core.utils import (
    classify_competition,
    parse_date_safe,
    fuzzy_name_search,
    TEAM_ALIASES,
)


def _normalize_calendar_name(raw: str, canonical_teams: list[str]) -> str:
    """Mapea un nombre de liga_calendar al nombre canonico corto del state."""
    match = fuzzy_name_search(raw, canonical_teams, TEAM_ALIASES)
    return match or raw


def _build_team_match_index(
    calendar_rows: list[dict],
    canonical_teams: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Construye {team: [matches sorted by date]} desde filas de liga_calendar.

    Incluye partidos tanto finalizados como programados, ya que ambos son
    necesarios para calcular dias_to_next y congestion futura.

    Los nombres de equipo se normalizan a los nombres canonicos cortos del state.
    """
    teams = canonical_teams or []
    index: dict[str, list[dict]] = defaultdict(list)
    for row in calendar_rows:
        md = parse_date_safe(row.get("match_date"))  # type: ignore[arg-type]
        if md is None:
            continue
        home_raw = (row.get("home_team") or "").strip()
        away_raw = (row.get("away_team") or "").strip()
        comp = classify_competition(row.get("competition", ""))
        status = row.get("status", "")
        home = _normalize_calendar_name(home_raw, teams) if teams else home_raw
        away = _normalize_calendar_name(away_raw, teams) if teams else away_raw
        entry_base = {"date": md, "competition": comp, "status": status}
        if home:
            index[home].append({**entry_base, "is_home": True, "opponent": away})
        if away:
            index[away].append({**entry_base, "is_home": False, "opponent": home})
    for team in index:
        index[team].sort(key=lambda m: m["date"])
    return dict(index)


def build_state(
    partidos: list[dict],
    *,
    objectives: dict | None = None,
    calendar_rows: list[dict] | None = None,
) -> dict:
    """Precalcula todo el estado estadistico necesario para generar features.

    Args:
        partidos:       lista de partidos crudos.
        objectives:     objetivos de La Liga (opcional, se omite si None).
        calendar_rows:  filas de liga_calendar para fatiga (opcional).

    Returns:
        Dict con partidos, scores, xstyles, ref_perfiles,
        perfiles_gmm, objectives, cal_index, updated_at.
    """
    scores = calcular_scores(partidos)
    xstyles = calcular_xstyle(partidos)
    ref_perfiles = calcular_perfiles_arbitros(partidos)
    perfiles_gmm = calcular_perfiles_gmm(partidos)

    canonical_teams = list(scores.keys()) if scores else None
    cal_index = (
        _build_team_match_index(calendar_rows, canonical_teams) if calendar_rows else {}
    )

    return {
        "partidos": partidos,
        "scores": scores,
        "xstyles": xstyles,
        "ref_perfiles": ref_perfiles,
        "perfiles_gmm": perfiles_gmm,
        "objectives": objectives or {},
        "cal_index": cal_index,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Singleton cache (para predicciones en vivo via generate.py)
# ---------------------------------------------------------------------------

_lock = Lock()
_cached: dict | None = None


def _load_from_supabase() -> dict:
    from selection import supabase_client

    partidos = supabase_client.fetch_all_matches()
    if not partidos:
        raise RuntimeError("No hay partidos disponibles en Supabase.")

    try:
        objectives = supabase_client.fetch_laliga_objectives()
    except Exception:
        logger.warning(
            "No se pudieron cargar los objectives de LaLiga; se usara dict vacio.",
            exc_info=True,
        )
        objectives = {}

    try:
        calendar_rows = supabase_client.fetch_liga_calendar()
    except Exception:
        logger.warning(
            "No se pudo cargar el calendario de liga; se omite cal_index.",
            exc_info=True,
        )
        calendar_rows = None

    return build_state(partidos, objectives=objectives, calendar_rows=calendar_rows)


def get_state(*, refresh: bool = False) -> dict:
    """Devuelve el estado cacheado. Carga de Supabase la primera vez o si refresh=True."""
    import copy

    global _cached
    with _lock:
        if _cached is None or refresh:
            _cached = _load_from_supabase()
        return copy.deepcopy(_cached)
