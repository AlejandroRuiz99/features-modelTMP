"""Estado estadistico precalculado (singleton thread-safe).

build_state()      — computa scores, rankings, xstyles, arbitros, GMM,
                      calendario y objectives a partir de una lista de
                      partidos (puro, sin I/O).
set_data_source()  — inyecta el callable que produce la lista de partidos
                      (composition root real, o un stub en tests).
get_state()        — devuelve el estado cacheado (zero-arg — design D1's
                      state_provider_fn contract); carga via el data source
                      inyectado la primera vez (o si refresh=True).

D1 (architecture-boundaries, REQ-12): esta capa leaf no debe depender de
`selection` (adaptador legacy de Supabase). La resolucion del data source
REAL (Supabase u otro) es responsabilidad de la composition root (ver
HWFP/cli/bot_main.py), no de este modulo.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

from HWFP.features.transformation import calcular_scores, calcular_xstyle
from HWFP.features.transformation.referees import calcular_perfiles as calcular_perfiles_arbitros
from HWFP.features.transformation.referee_gmm import calcular_perfiles_gmm
from HWFP.features.core.utils import (
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
# Singleton cache with an injectable data source (composition-root DI point)
# ---------------------------------------------------------------------------

DataSourceFn = Callable[[], list[dict]]

_lock = Lock()
_cached: dict | None = None
_data_source: DataSourceFn | None = None


def set_data_source(fn: DataSourceFn) -> None:
    """Injects the callable that produces the raw `partidos` list.

    The composition root wires the real adapter here (e.g. a Supabase-backed
    fetcher); tests wire a stub. This keeps the leaf `HWFP.features` package
    free of any dependency on a concrete data-fetching implementation
    (REQ-12/14, architecture-boundaries).
    """
    global _data_source
    _data_source = fn


def _load_state() -> dict:
    if _data_source is None:
        raise RuntimeError(
            "No data source configured for HWFP.features.core.state_cache. "
            "Call set_data_source(fn) from the composition root before "
            "get_state() is invoked."
        )
    partidos = _data_source()
    if not partidos:
        raise RuntimeError("El data source configurado no devolvio partidos.")

    # D17: objectives are injected per-match via overlay (narrative YAML),
    # not fetched here. Calendar rows (multi-competition fatigue) are an
    # optional enrichment the composition root's data source may bundle in
    # the future; this leaf module does not fetch them itself.
    return build_state(partidos, objectives={}, calendar_rows=None)


def get_state(*, refresh: bool = False) -> dict:
    """Returns the cached state (zero-arg call — design D1's state_provider_fn contract).

    Loads via the injected data source the first time, or whenever
    refresh=True. Call set_data_source() before the first invocation.
    """
    import copy

    global _cached
    with _lock:
        if _cached is None or refresh:
            _cached = _load_state()
        return copy.deepcopy(_cached)
