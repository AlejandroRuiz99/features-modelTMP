"""Conexión y operaciones CRUD contra Supabase."""

from __future__ import annotations

import logging

from supabase import create_client, Client

logger = logging.getLogger(__name__)

from core.config import SUPABASE_URL, SUPABASE_KEY


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _row_to_partido(row: dict) -> dict:
    """Convierte una fila de Supabase al formato interno del modelo."""
    return {
        "fixture_id": row["match_id"],
        "date": row["match_date"],
        "season": int(row["season"].split("-")[0]),
        "jornada": None,
        "referee": row.get("referee") or "",
        "home": {
            "name": row["home_team"],
            "fouls": row.get("fouls_home") or 0,
            "yellow_cards": row.get("yellows_home") or 0,
            "red_cards": row.get("reds_home") or 0,
            "shots": row.get("shots_home") or 0,
            "shots_on_target": row.get("shots_on_target_home") or 0,
            "corners": row.get("corners_home") or 0,
            "goals": row.get("home_goals") or 0,
            "possession": row.get("possession_home"),
        },
        "away": {
            "name": row["away_team"],
            "fouls": row.get("fouls_away") or 0,
            "yellow_cards": row.get("yellows_away") or 0,
            "red_cards": row.get("reds_away") or 0,
            "shots": row.get("shots_away") or 0,
            "shots_on_target": row.get("shots_on_target_away") or 0,
            "corners": row.get("corners_away") or 0,
            "goals": row.get("away_goals") or 0,
            "possession": row.get("possession_away"),
        },
        "odds": {
            "b365_home": row.get("odds_b365_home"),
            "b365_draw": row.get("odds_b365_draw"),
            "b365_away": row.get("odds_b365_away"),
            "ou25_over": row.get("odds_ou25_over"),
            "ou25_under": row.get("odds_ou25_under"),
        },
    }


def _paginate_select(
    sb: Client, table: str, columns: str, *, query_fn=None, **kwargs
) -> list[dict]:
    """Descarga todas las filas de una tabla con paginacion automatica.

    Args:
        query_fn: Callable(query) -> query para aplicar filtros adicionales
                  (ej: lambda q: q.not_.is_("col", "null")).
    """
    all_rows: list[dict] = []
    page_size = 1000
    offset = 0
    order_col = kwargs.get("order")
    order_desc = kwargs.get("desc", False)

    while True:
        q = sb.table(table).select(columns)
        if query_fn is not None:
            q = query_fn(q)
        if order_col:
            q = q.order(order_col, desc=order_desc)
        r = q.range(offset, offset + page_size - 1).execute()
        all_rows.extend(r.data)
        if len(r.data) < page_size:
            break
        offset += page_size

    return all_rows


def fetch_all_matches() -> list[dict]:
    """Descarga todos los partidos de Supabase → formato modelo."""
    sb = get_client()
    cols = (
        "match_id,match_date,season,referee,"
        "home_team,away_team,home_goals,away_goals,"
        "fouls_home,fouls_away,yellows_home,yellows_away,"
        "reds_home,reds_away,shots_home,shots_away,"
        "shots_on_target_home,shots_on_target_away,"
        "corners_home,corners_away,"
        "possession_home,possession_away,"
        "odds_b365_home,odds_b365_draw,odds_b365_away,"
        "odds_ou25_over,odds_ou25_under"
    )
    rows = _paginate_select(sb, "matches", cols, order="match_date")
    return [_row_to_partido(r) for r in rows]


def get_existing_keys(sb: Client) -> set[tuple[str, str, str]]:
    """Devuelve conjunto de (date, home, away) ya existentes."""
    rows = _paginate_select(sb, "matches", "match_date,home_team,away_team")
    return {(r["match_date"], r["home_team"], r["away_team"]) for r in rows}


def sync_new_matches(records: list[dict]) -> tuple[int, int]:
    """Sube partidos nuevos a Supabase. Devuelve (insertados, omitidos)."""
    sb = get_client()

    logger.info("Consultando partidos existentes en Supabase...")
    existing = get_existing_keys(sb)
    logger.info("%d partidos ya en Supabase.", len(existing))

    nuevos = [
        r
        for r in records
        if (r["match_date"], r["home_team"], r["away_team"]) not in existing
    ]

    if not nuevos:
        return 0, len(records)

    batch_size = 500
    insertados = 0
    for i in range(0, len(nuevos), batch_size):
        batch = nuevos[i : i + batch_size]
        sb.table("matches").insert(batch).execute()
        insertados += len(batch)

    return insertados, len(records) - insertados


def get_matches_with_possession() -> set[tuple[str, str, str]]:
    """Devuelve conjunto de (date, home, away) que ya tienen posesion registrada."""
    sb = get_client()
    rows = _paginate_select(
        sb,
        "matches",
        "match_date,home_team,away_team",
        query_fn=lambda q: q.not_.is_("possession_home", "null"),
    )
    return {(r["match_date"], r["home_team"], r["away_team"]) for r in rows}


def update_possession_single(
    match_date: str,
    home_team: str,
    away_team: str,
    possession_home: int,
    possession_away: int,
) -> bool:
    """Actualiza posesión de un partido concreto. Devuelve True si se actualizó."""
    sb = get_client()
    try:
        result = (
            sb.table("matches")
            .update(
                {
                    "possession_home": possession_home,
                    "possession_away": possession_away,
                }
            )
            .eq("match_date", match_date)
            .eq("home_team", home_team)
            .eq("away_team", away_team)
            .execute()
        )
        return bool(result.data)
    except Exception:
        logger.warning(
            "Error actualizando posesion para %s vs %s (%s)",
            home_team,
            away_team,
            match_date,
            exc_info=True,
        )
        return False


def fetch_laliga_objectives() -> dict[str, dict]:
    """
    Descarga la tabla laliga_objectives y devuelve un dict indexado por equipo.

    Retorna:
        {
          "Barcelona": {
            "categoria": "Título",
            "num_categoria": 1,
            "objetivo_label": "titulo",
            "competiciones_activas": True,
          },
          ...
        }
    """
    _LABEL_MAP = {
        1: "titulo",
        2: "top4",
        3: "uel",
        4: "uecl",
        5: "media_tabla",
        6: "descenso",
    }
    sb = get_client()
    rows = (
        sb.table("laliga_objectives")
        .select("equipo,categoria,num_categoria,competiciones_activas")
        .execute()
    )
    result: dict[str, dict] = {}
    for r in rows.data or []:
        equipo = (r.get("equipo") or "").strip()
        if not equipo:
            continue
        num = int(r.get("num_categoria") or 5)
        result[equipo] = {
            "categoria": r.get("categoria", ""),
            "num_categoria": num,
            "objetivo_label": _LABEL_MAP.get(num, "media_tabla"),
            "competiciones_activas": bool(r.get("competiciones_activas", False)),
        }
    return result


def fetch_liga_calendar() -> list[dict]:
    """
    Descarga toda la tabla liga_calendar (partidos multi-competición de equipos
    de La Liga: Liga, Copa del Rey, UCL, UEL, UECL).

    Retorna lista de dicts con: home_team, away_team, match_date, competition,
    season, status, round, etc.
    """
    sb = get_client()
    rows = _paginate_select(
        sb,
        "liga_calendar",
        "home_team,away_team,match_date,match_time,competition,season,status,round",
        order="match_date",
    )
    return rows


def fetch_latest_odds_rows(sport: str = "soccer") -> dict:
    """
    Devuelve todas las filas de odds_raw para el último scraped_at.

    Retorna:
      {
        "scraped_at": str | None,
        "rows": [row...]
      }
    """
    sb = get_client()
    latest_resp = (
        sb.table("odds_raw")
        .select("scraped_at")
        .order("scraped_at", desc=True)
        .limit(1)
        .execute()
    )
    if not latest_resp.data:
        return {"scraped_at": None, "rows": []}

    scraped_at = latest_resp.data[0].get("scraped_at")
    if not scraped_at:
        return {"scraped_at": None, "rows": []}

    page_size = 1000
    offset = 0
    rows: list[dict] = []
    while True:
        q = (
            sb.table("odds_raw")
            .select(
                "home_team,away_team,mercado,selection,cuota,external_event_id,partido,sport,scraped_at"
            )
            .eq("scraped_at", scraped_at)
            .range(offset, offset + page_size - 1)
        )
        if sport:
            q = q.eq("sport", sport)
        r = q.execute()
        rows.extend(r.data)
        if len(r.data) < page_size:
            break
        offset += page_size

    return {"scraped_at": scraped_at, "rows": rows}
