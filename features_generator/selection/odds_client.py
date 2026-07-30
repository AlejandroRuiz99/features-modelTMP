"""Fetch de cuotas de apuestas desde Supabase."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from selection import supabase_client
from HWFP.features.transformation import buscar_equipo
from HWFP.features.core.utils import team_match


def get_match_odds_rows(
    scores: dict[str, Any],
    equipo_local: str,
    equipo_visitante: str,
    *,
    match_date: str | None = None,
    window_days_before: int = 4,
    window_days_after: int = 1,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Busca las filas de cuotas para un partido concreto en Supabase.

    Args:
        scores: Diccionario de scores del state (para resolver aliases de equipo).
        equipo_local: Nombre canónico del equipo local.
        equipo_visitante: Nombre canónico del equipo visitante.
        match_date: Fecha del partido (ISO 'YYYY-MM-DD'). Si se pasa, la
            búsqueda usa `fetch_odds_rows_for_match_window()` que mira todas
            las filas cuyo scraped_at cae en la ventana
            [match_date - window_days_before, match_date + window_days_after].
            Si es None, se usa el comportamiento legacy (solo el último
            scrape global vía `fetch_latest_odds_rows`), que puede perder
            partidos cuando los scrapes son diarios.
        window_days_before: Días antes del partido incluidos en la ventana
            cuando `match_date` no es None (por defecto 4).
        window_days_after: Días después del partido incluidos en la ventana
            cuando `match_date` no es None (por defecto 1).

    Returns:
        (rows_filtradas, scraped_at, event_id)

        `scraped_at` es el scraped_at del scrape usado (latest global si
        `match_date` es None, o el `latest_scraped_at` dentro de la ventana
        si `match_date` está presente).
    """
    if match_date is not None:
        odds = supabase_client.fetch_odds_rows_for_match_window(
            match_date,
            sport="soccer",
            days_before=window_days_before,
            days_after=window_days_after,
        )
        rows = odds.get("rows", [])
        scraped_at = odds.get("latest_scraped_at")
    else:
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

    # Un mismo partido puede aparecer en odds_raw con MÚLTIPLES event_ids
    # por dos motivos:
    #   (a) scrapes diarios del mismo evento (viernes + sábado + domingo)
    #   (b) fuentes mixtas: scrape automático de Codere + uploads manuales
    #       de mercados específicos (p.ej. líneas de faltas cargadas a mano)
    #
    # La estrategia correcta es UNIFICAR todas las filas del partido y
    # deduplicar por (mercado, selection) quedándose con la cuota más
    # reciente (mayor scraped_at). Esto preserva el universo completo de
    # mercados disponibles sin duplicar cuotas cuando las mismas líneas
    # aparecen en varios scrapes.
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    event_ids_seen: dict[str, int] = defaultdict(int)
    for r in match_rows:
        mercado = (r.get("mercado") or "").strip()
        selection = (r.get("selection") or "").strip()
        if not mercado or not selection:
            continue
        key = (mercado, selection)
        existing = dedup.get(key)
        if existing is None or (r.get("scraped_at") or "") > (
            existing.get("scraped_at") or ""
        ):
            dedup[key] = r
        ev = (r.get("external_event_id") or "").strip()
        if ev:
            event_ids_seen[ev] += 1

    match_rows = list(dedup.values())

    # Diagnóstico: el event_id "principal" reportado es el que más filas
    # contribuyó (útil para trazas/logs; no afecta al fetch).
    event_id: str | None = None
    if event_ids_seen:
        event_id = max(event_ids_seen.items(), key=lambda x: x[1])[0]

    return match_rows, scraped_at, event_id
