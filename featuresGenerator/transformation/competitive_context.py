"""Contexto competitivo de un equipo antes de un partido.

Encapsula en un unico modulo:
  - Tabla de clasificacion (calculada desde partidos de la temporada)
  - Objetivo de liga y urgencia (solo desde laliga_objectives)
  - Fatiga multi-competicion (implementacion privada, antes en fatigue.py)
  - Forma reciente (ultimos n partidos de Liga)
  - ICC — Indice de Competitividad Contextual
  - Interaccion historica arbitro-equipo

Principio: NUNCA inferir lo que no se conoce.
Si un equipo no esta en laliga_objectives, el ICC y la urgencia son None
y los factores de ajuste son neutros (1.0). Solo se computan hechos
observables: tabla, forma, fatiga y calendario.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from core.utils import norm_text, parse_date_safe


# ---------------------------------------------------------------------------
# Fatiga — implementacion privada (antes stats/fatigue.py)
# ---------------------------------------------------------------------------

_FATIGUE_TAU = 4.5  # dias; decaimiento exponencial base


def _compute_fatigue_score(
    days_since_last: Optional[int],
    *,
    matches_last_14d: int = 2,
    last_is_away: bool = False,
    last_competition: str = "liga",
) -> float:
    """Score de fatiga [0, 1]. 0 = descansado, 1 = agotado."""
    if days_since_last is None or days_since_last < 0:
        base = 0.30
    else:
        base = math.exp(-days_since_last / _FATIGUE_TAU)

    base += max(0, matches_last_14d - 2) * 0.06

    if last_is_away and last_competition in ("ucl", "uel", "uecl"):
        base += 0.08

    return round(max(0.0, min(1.0, base)), 4)


# ---------------------------------------------------------------------------
# Tabla de clasificacion
# ---------------------------------------------------------------------------

def _current_season_games(state: dict) -> tuple[int, list[dict]]:
    """Temporada actual y sus partidos ordenados cronologicamente."""
    partidos = state["partidos"]
    season = max(
        (p.get("season") for p in partidos if p.get("season") is not None),
        default=2025,
    )
    return season, sorted(
        [p for p in partidos if p.get("season") == season],
        key=lambda p: p["date"],
    )


def _build_league_table(season_games: list[dict]) -> dict[str, dict]:
    """Tabla de clasificacion calculada desde resultados reales."""
    tabla: dict[str, dict] = {}
    for p in season_games:
        h = p["home"]["name"]
        a = p["away"]["name"]
        hg = p["home"].get("goals", 0) or 0
        ag = p["away"].get("goals", 0) or 0
        for team in (h, a):
            if team not in tabla:
                tabla[team] = {"points": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0}
        tabla[h]["gf"] += hg; tabla[h]["ga"] += ag; tabla[h]["gd"] = tabla[h]["gf"] - tabla[h]["ga"]; tabla[h]["played"] += 1
        tabla[a]["gf"] += ag; tabla[a]["ga"] += hg; tabla[a]["gd"] = tabla[a]["gf"] - tabla[a]["ga"]; tabla[a]["played"] += 1
        if hg > ag:   tabla[h]["points"] += 3
        elif hg < ag: tabla[a]["points"] += 3
        else:         tabla[h]["points"] += 1; tabla[a]["points"] += 1

    for pos, (team, data) in enumerate(
        sorted(tabla.items(), key=lambda kv: (kv[1]["points"], kv[1]["gd"], kv[1]["gf"]), reverse=True),
        start=1,
    ):
        data["position"] = pos
    return tabla


# ---------------------------------------------------------------------------
# Forma reciente
# ---------------------------------------------------------------------------

def _calc_recent_form(season_games: list[dict], team_name: str, n: int = 5) -> dict:
    """Ultimos n partidos de Liga: puntos, secuencia W/D/L, momentum."""
    games = sorted(
        [p for p in season_games
         if p["home"]["name"] == team_name or p["away"]["name"] == team_name],
        key=lambda p: p["date"], reverse=True,
    )[:n]

    pts = 0
    seq = []
    for p in games:
        is_home = p["home"]["name"] == team_name
        gf = (p["home"] if is_home else p["away"]).get("goals", 0) or 0
        ga = (p["away"] if is_home else p["home"]).get("goals", 0) or 0
        if gf > ga:   pts += 3; seq.append("W")
        elif gf == ga: pts += 1; seq.append("D")
        else:          seq.append("L")

    ppg = round(pts / max(1, len(games)), 2)
    return {
        "matches":        len(games),
        "points":         pts,
        "ppg":            ppg,
        "sequence":       "".join(seq),
        "momentum_score": round(min(1.0, ppg / 3.0), 3),
    }


# ---------------------------------------------------------------------------
# Calendario y fatiga del equipo
# ---------------------------------------------------------------------------

def _extract_calendar_values(
    cal_index: dict[str, list[dict]],
    season_games: list[dict],
    team_name: str,
    match_date: Optional[str] = None,
) -> dict:
    """Extrae valores de calendario para el calculo de fatiga.

    Fuente principal: cal_index (tabla liga_calendar de Supabase), que incluye
    partidos multi-competicion (Liga, UCL, Copa, UEL, UECL).
    Fallback: partidos de Liga en season_games (solo Liga).
    """
    from datetime import timedelta

    ref_date = parse_date_safe(match_date) if match_date else datetime.now().date()
    if ref_date is None:
        ref_date = datetime.now().date()

    team_cal = cal_index.get(team_name, [])
    if team_cal:
        past = [m for m in team_cal if m["date"] < ref_date]
        future = [m for m in team_cal if m["date"] > ref_date]

        days_since_last = None
        last_is_away = False
        last_competition = "liga"
        if past:
            last_match = past[-1]
            days_since_last = (ref_date - last_match["date"]).days
            last_is_away = not last_match["is_home"]
            last_competition = last_match.get("competition", "liga")

        days_to_next = None
        next_competition = "liga"
        next_is_home = True
        if future:
            next_match = future[0]
            days_to_next = (next_match["date"] - ref_date).days
            next_competition = next_match.get("competition", "liga")
            next_is_home = next_match["is_home"]

        window_past = ref_date - timedelta(days=14)
        window_future = ref_date + timedelta(days=14)
        matches_last_14d = sum(1 for m in past if m["date"] >= window_past)
        matches_next_14d = sum(1 for m in future if m["date"] <= window_future)

        congestion = round(min(1.0, max(0.0, (matches_last_14d - 1) * 0.20)), 3)

        return {
            "days_since_last":  days_since_last,
            "days_to_next":     days_to_next,
            "last_competition": last_competition,
            "next_competition": next_competition,
            "last_is_away":     last_is_away,
            "matches_last_14d": matches_last_14d,
            "matches_next_14d": matches_next_14d,
            "congestion_score": congestion,
            "source":           "supabase_calendar",
        }

    # Fallback: solo Liga (season_games no tiene datos multi-competicion)
    team_games = sorted(
        [p for p in season_games
         if p["home"]["name"] == team_name or p["away"]["name"] == team_name],
        key=lambda p: p["date"], reverse=True,
    )

    days_since_last = None
    last_is_away = False
    matches_last_14d = 2

    if team_games:
        last_date = parse_date_safe(team_games[0]["date"])
        if last_date:
            days_since_last = max(0, (ref_date - last_date).days)
            last_is_away = team_games[0]["away"]["name"] == team_name
            matches_last_14d = sum(
                1 for g in team_games
                if (d := parse_date_safe(g["date"])) and (ref_date - d).days <= 14
            )

    return {
        "days_since_last":  days_since_last,
        "days_to_next":     None,
        "last_competition": "liga",
        "next_competition": "liga",
        "last_is_away":     last_is_away,
        "matches_last_14d": matches_last_14d,
        "matches_next_14d": 1,
        "congestion_score": round(min(1.0, max(0.0, (matches_last_14d - 1) * 0.20)), 3),
        "source":           "supabase_liga_only",
    }


# ---------------------------------------------------------------------------
# Objetivo de liga (solo desde laliga_objectives, sin inferencia)
# ---------------------------------------------------------------------------

_URGENCY_BY_OBJECTIVE: dict[int, float] = {
    1: 0.85,   # Titulo
    2: 0.75,   # Champions (top4)
    3: 0.60,   # Europa League
    4: 0.45,   # Conference
    5: 0.25,   # Media tabla
    6: 0.80,   # Descenso
}

_TARGET_POSITIONS: dict[str, int] = {
    "titulo": 2, "top4": 4, "uel": 6, "descenso": 18, "salvacion": 15,
}


def _get_objective(
    team_name: str,
    objectives: dict[str, dict],
) -> tuple[Optional[str], Optional[float], bool]:
    """Lee objetivo desde laliga_objectives.

    Returns (objetivo_label, urgency_base, competiciones_activas).
    Si el equipo no esta en la tabla: (None, None, False).
    Sin fallback posicional.
    """
    obj = objectives.get(team_name)
    if not obj:
        return None, None, False
    return (
        obj.get("objetivo_label"),
        _URGENCY_BY_OBJECTIVE.get(int(obj.get("num_categoria", 5)), 0.25),
        bool(obj.get("competiciones_activas", False)),
    )


def _calc_objective_pressure(
    objetivo_label: str,
    table: dict[str, dict],
    team_name: str,
) -> tuple[float, dict]:
    """Presion real de tabla: cercania al objetivo declarado en puntos."""
    ref_pos = _TARGET_POSITIONS.get(objetivo_label)
    me = table.get(team_name, {})
    my_pts = me.get("points", 0)

    if not ref_pos:
        return 0.35, {"target_position": None, "points_gap": None}

    candidates = [d for d in table.values() if d.get("position") == ref_pos]
    target_pts = candidates[0]["points"] if candidates else my_pts
    gap = abs(my_pts - target_pts)
    closeness = max(0.0, 1.0 - min(gap, 12) / 12.0)
    base = 0.55 if objetivo_label in {"top4", "uel", "salvacion"} else 0.70
    return round(min(1.0, max(0.0, 0.4 * base + 0.6 * closeness)), 3), {
        "target_position": ref_pos,
        "points_gap":      gap,
    }


# ---------------------------------------------------------------------------
# Riesgo de rotacion (basado en calendario; solo si hay datos)
# ---------------------------------------------------------------------------

def _calc_rotation_score(
    days_to_next: Optional[float],
    next_competition: str,
    competiciones_activas: bool,
) -> float:
    """Score de rotacion [0, 1] basado en carga proxima de partidos."""
    score = 0.5
    if competiciones_activas:
        score = min(1.0, score + 0.15)
    if isinstance(days_to_next, (int, float)):
        if days_to_next <= 3:
            score = min(1.0, score + 0.15)
        elif days_to_next >= 7:
            score = max(0.0, score - 0.10)
        if next_competition in ("ucl", "uel") and days_to_next <= 4:
            score = min(1.0, score + 0.10)
    return round(score, 3)


# ---------------------------------------------------------------------------
# ICC — Indice de Competitividad Contextual
# ---------------------------------------------------------------------------

def _calc_icc(
    urgency_score: Optional[float],
    momentum: float,
    fatigue_score: float,
    rotation_score: float,
) -> Optional[float]:
    """ICC en [-1, 1]. None si no hay datos de urgencia (sin laliga_objectives)."""
    if urgency_score is None:
        return None
    icc = (
        0.40 * urgency_score
        + 0.25 * momentum
        - 0.20 * fatigue_score
        - 0.15 * rotation_score
    )
    return round(max(-1.0, min(1.0, icc)), 3)


def _icc_label(icc: Optional[float]) -> Optional[str]:
    if icc is None:
        return None
    return "alto" if icc >= 0.2 else "bajo" if icc <= -0.2 else "medio"


def _urgencia_label(urgency_score: Optional[float]) -> Optional[str]:
    if urgency_score is None:
        return None
    return "alta" if urgency_score >= 0.7 else "media" if urgency_score >= 0.45 else "baja"


# ---------------------------------------------------------------------------
# Interaccion historica arbitro-equipo
# ---------------------------------------------------------------------------

def _same_referee(a: Optional[str], b: Optional[str]) -> bool:
    na, nb = norm_text(a), norm_text(b)
    return bool(na and nb and (na == nb or na in nb or nb in na))


def _fouls_in_match(match: dict, team_name: str) -> Optional[float]:
    if norm_text(match.get("home", {}).get("name")) == norm_text(team_name):
        return float(match.get("home", {}).get("fouls", 0) or 0)
    if norm_text(match.get("away", {}).get("name")) == norm_text(team_name):
        return float(match.get("away", {}).get("fouls", 0) or 0)
    return None


def _calc_ref_team_delta(
    *,
    partidos: list[dict],
    referee_name: Optional[str],
    team_name: str,
    shrinkage_k: float = 8.0,
    min_samples: int = 2,
) -> tuple[float, int]:
    """Delta de faltas cometidas por este equipo cuando arbitra este arbitro."""
    if not referee_name:
        return 0.0, 0

    baseline, pair = [], []
    for p in partidos:
        v = _fouls_in_match(p, team_name)
        if v is None:
            continue
        baseline.append(v)
        if _same_referee(p.get("referee"), referee_name):
            pair.append(v)

    if not baseline or len(pair) < min_samples:
        return 0.0, 0

    n = len(pair)
    delta = max(-4.0, min(4.0, sum(pair) / n - sum(baseline) / len(baseline)))
    return round(delta * n / (n + shrinkage_k), 3), n


def _build_referee_interaction(
    *,
    partidos: list[dict],
    referee_name: Optional[str],
    eq_local: str,
    eq_visit: str,
) -> dict:
    d_local, n_local = _calc_ref_team_delta(
        partidos=partidos, referee_name=referee_name, team_name=eq_local,
    )
    d_visit, n_visit = _calc_ref_team_delta(
        partidos=partidos, referee_name=referee_name, team_name=eq_visit,
    )
    return {
        "delta_local":          d_local,
        "delta_visitante":      d_visit,
        "delta_sum":            round(d_local + d_visit, 3),
        "n_partidos_local":     n_local,
        "n_partidos_visitante": n_visit,
    }


# ---------------------------------------------------------------------------
# Contexto de equipo
# ---------------------------------------------------------------------------

def _build_team_context(
    *,
    season_games: list[dict],
    table: dict[str, dict],
    team_name: str,
    cal_index: dict[str, list[dict]],
    objectives: dict[str, dict],
    match_date: Optional[str] = None,
) -> tuple[dict, dict]:
    """Compone el contexto competitivo de un equipo.

    Returns:
        (context_payload, model_input)
    """
    # Datos facticos — siempre disponibles
    table_entry = table.get(team_name, {})
    cal = _extract_calendar_values(cal_index, season_games, team_name, match_date)
    forma = _calc_recent_form(season_games, team_name, n=5)
    fatigue_score = _compute_fatigue_score(
        cal["days_since_last"],
        matches_last_14d=cal["matches_last_14d"],
        last_is_away=cal["last_is_away"],
        last_competition=cal["last_competition"],
    )

    # Datos de objetivo — solo desde laliga_objectives, sin inferencia
    objetivo_label, urgency_base, competiciones_activas = _get_objective(team_name, objectives)

    if urgency_base is not None:
        pressure_score, pressure_meta = _calc_objective_pressure(objetivo_label, table, team_name)
        urgency_score = round(0.60 * urgency_base + 0.40 * pressure_score, 3)
        rotation_score = _calc_rotation_score(
            cal["days_to_next"], cal["next_competition"], competiciones_activas,
        )
    else:
        urgency_score = None
        pressure_meta = {"target_position": None, "points_gap": None}
        rotation_score = _calc_rotation_score(cal["days_to_next"], cal["next_competition"], False)

    icc = _calc_icc(urgency_score, forma["momentum_score"], fatigue_score, rotation_score)

    # Factores de ajuste: neutros (1.0 / 0.0) si no hay ICC
    factors = {
        "xg_factor":         round(1.0 + 0.10 * icc, 4) if icc is not None else 1.0,
        "xfouls_factor":     round(1.0 + 0.12 * icc, 4) if icc is not None else 1.0,
        "posesion_delta_pp": round(2.0 * icc, 3) if icc is not None else 0.0,
    }

    context = {
        "tabla": {
            "position":  table_entry.get("position"),
            "points":    table_entry.get("points"),
            "played":    table_entry.get("played"),
            "goal_diff": table_entry.get("gd"),
        },
        "forma_reciente": {
            "window_matches": forma["matches"],
            "points":         forma["points"],
            "ppg":            forma["ppg"],
            "sequence":       forma["sequence"],
            "momentum_score": forma["momentum_score"],
        },
        "calendario": {
            "days_since_last":  cal["days_since_last"],
            "days_to_next":     cal["days_to_next"],
            "last_competition": cal["last_competition"],
            "next_competition": cal["next_competition"],
            "last_is_away":     cal["last_is_away"],
            "matches_last_14d": cal["matches_last_14d"],
            "congestion_score": cal["congestion_score"],
            "source":           cal["source"],
        },
        "competitividad": {
            "objetivo_liga":         objetivo_label,
            "liga_urgencia":         _urgencia_label(urgency_score),
            "competiciones_activas": competiciones_activas,
            "urgency_score":         urgency_score,
            "fatigue_score":         round(fatigue_score, 3),
            "rotation_score":        round(rotation_score, 3),
            "icc_score":             icc,
            "icc_label":             _icc_label(icc),
            "pressure_meta":         pressure_meta,
            "factors":               factors,
        },
    }

    model_input = {
        "days_since_last":       cal["days_since_last"],
        "days_to_next":          cal["days_to_next"],
        "last_competition":      cal["last_competition"],
        "next_competition":      cal["next_competition"],
        "last_is_away":          cal["last_is_away"],
        "matches_last_14d":      cal["matches_last_14d"],
        "liga_urgencia":         _urgencia_label(urgency_score),
        "objetivo_liga":         objetivo_label,
        "competiciones_activas": competiciones_activas,
    }

    return context, model_input


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def build_context_payload(
    *,
    state: dict[str, Any],
    eq_local: str,
    eq_visit: str,
    jornada: int,
    contexto_temporada: dict[str, Any],
    arbitro: Optional[str],
    arbitraje_source: str,
    match_date: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    season, season_games = _current_season_games(state)
    table = _build_league_table(season_games)
    objectives = state.get("objectives") or {}
    cal_index = state.get("cal_index") or {}

    local_ctx, local_model = _build_team_context(
        season_games=season_games, table=table, team_name=eq_local,
        cal_index=cal_index, objectives=objectives, match_date=match_date,
    )
    visit_ctx, visit_model = _build_team_context(
        season_games=season_games, table=table, team_name=eq_visit,
        cal_index=cal_index, objectives=objectives, match_date=match_date,
    )

    payload = {
        "temporada": {
            "season":             f"{season}-{(season + 1) % 100:02d}",
            "jornada":            jornada,
            "tramo":              contexto_temporada.get("tramo"),
            "jornadas_restantes": contexto_temporada.get("jornadas_restantes"),
        },
        "competitivo": {
            "local":     local_ctx,
            "visitante": visit_ctx,
        },
        "arbitraje": {
            "arbitro": arbitro,
            "source":  arbitraje_source,
            "interaccion_equipos": _build_referee_interaction(
                partidos=state["partidos"],
                referee_name=arbitro,
                eq_local=eq_local,
                eq_visit=eq_visit,
            ),
        },
    }

    return payload, {"local": local_model, "visitante": visit_model}
