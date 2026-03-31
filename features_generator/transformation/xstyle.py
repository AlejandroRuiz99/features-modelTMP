"""xStyle — Perfil de estilo de juego de cada equipo.

Dimensiones: posesión, tiros, precisión, corners, goles, eficiencia,
             físico, riesgo tarjeta, faltas provocadas.
"""

from __future__ import annotations

from datetime import date
from core.config import DECAY_LAMBDA
from core.helpers import decay_weight, parse_date, safe


# ---------------------------------------------------------------------------
# Clasificacion de estilo
# ---------------------------------------------------------------------------

def _clasificar_estilo(
    fouls: float,
    shots: float,
    shot_acc: float,
    set_piece_ratio: float,
    goals: float,
    goals_conceded: float,
) -> tuple[str, str]:
    """Clasifica el estilo segun metricas combinadas."""
    high_physical  = fouls > 13.5
    low_physical   = fouls < 11.0
    high_shots     = shots > 12.0
    low_shots      = shots < 9.5
    high_acc       = shot_acc > 0.40
    high_set_piece = set_piece_ratio > 0.33
    high_goals     = goals > 1.4

    if high_physical and low_shots:                      return "FÍSICO-DEFENSIVO",  "Alta presión, bajo volumen ofensivo"
    if high_physical and high_shots:                     return "INTENSO",            "Máxima intensidad en ambas fases"
    if low_physical  and high_shots and high_acc:        return "TÉCNICO-OFENSIVO",  "Juego de posesión con alto aprovechamiento"
    if low_physical  and high_shots:                     return "POSESIÓN",           "Control del juego con volumen ofensivo"
    if high_set_piece and high_physical:                 return "DIRECTO-FÍSICO",    "Balón parado y contacto físico constante"
    if high_set_piece:                                   return "ESTRATÉGICO",        "Dependencia de estrategia y balones parados"
    if low_physical  and low_shots:                      return "CONSERVADOR",        "Bajo riesgo, contención defensiva"
    if high_goals    and high_shots:                     return "OFENSIVO",           "Alto volumen goleador"
    return "EQUILIBRADO", "Sin tendencia dominante marcada"


# ---------------------------------------------------------------------------
# Acumulador por equipo (NamedTuple mutable → usamos dict con nombres claros)
# ---------------------------------------------------------------------------

def _zero_accum() -> dict:
    return {
        "weighted_fouls":          0.0,
        "weighted_yellows":        0.0,
        "weighted_reds":           0.0,
        "weighted_shots":          0.0,
        "weighted_shots_on_target": 0.0,
        "weighted_corners":        0.0,
        "weighted_goals_for":      0.0,
        "weighted_goals_against":  0.0,
        "weighted_fouls_suffered": 0.0,
        "total_weight":            0.0,
        "poss_weight_sum":         0.0,
        "poss_weight_total":       0.0,
        "n_matches":               0,
    }


# ---------------------------------------------------------------------------
# Calculo principal
# ---------------------------------------------------------------------------

def calcular_xstyle(partidos: list[dict]) -> dict:
    """Calcula el perfil de estilo de juego de cada equipo con decay temporal."""
    hoy = date.today()
    accum: dict[str, dict] = {}

    for p in partidos:
        peso = decay_weight(parse_date(p["date"]), hoy, DECAY_LAMBDA)

        for team_key, opp_key in (("home", "away"), ("away", "home")):
            team = p[team_key]
            opp  = p[opp_key]
            name = team["name"]

            if name not in accum:
                accum[name] = _zero_accum()
            a = accum[name]

            a["weighted_fouls"]           += safe(team.get("fouls"))          * peso
            a["weighted_yellows"]         += safe(team.get("yellow_cards"))    * peso
            a["weighted_reds"]            += safe(team.get("red_cards"))       * peso
            a["weighted_shots"]           += safe(team.get("shots"))           * peso
            a["weighted_shots_on_target"] += safe(team.get("shots_on_target")) * peso
            a["weighted_corners"]         += safe(team.get("corners"))         * peso
            a["weighted_goals_for"]       += safe(team.get("goals"))           * peso
            a["weighted_goals_against"]   += safe(opp.get("goals"))            * peso
            a["weighted_fouls_suffered"]  += safe(opp.get("fouls"))            * peso
            a["total_weight"]             += peso
            a["n_matches"]                += 1

            poss = team.get("possession")
            if poss is not None:
                a["poss_weight_sum"]   += float(poss) * peso
                a["poss_weight_total"] += peso

    profiles = {name: _build_profile(a) for name, a in accum.items() if a["total_weight"] > 0}
    _normalizar_dims(profiles)
    return profiles


def _build_profile(a: dict) -> dict:
    w = a["total_weight"]

    fouls    = a["weighted_fouls"]           / w
    yellows  = a["weighted_yellows"]         / w
    shots    = a["weighted_shots"]           / w
    shots_ot = a["weighted_shots_on_target"] / w
    corners  = a["weighted_corners"]         / w
    goals    = a["weighted_goals_for"]       / w
    goals_c  = a["weighted_goals_against"]   / w
    suffered = a["weighted_fouls_suffered"]  / w

    precision      = shots_ot / shots         if shots         > 0 else 0.0
    eficiencia     = goals    / shots_ot      if shots_ot      > 0 else 0.0
    set_piece_r    = corners  / (shots + corners) if (shots + corners) > 0 else 0.0
    cards_per_foul = yellows  / fouls         if fouls         > 0 else 0.0
    ratio_fisico   = fouls    / (fouls + shots) if (fouls + shots) > 0 else 0.5
    tempo          = fouls + shots + corners * 0.5

    poss_avg = (
        a["poss_weight_sum"] / a["poss_weight_total"]
        if a["poss_weight_total"] > 0 else None
    )

    estilo, estilo_desc = _clasificar_estilo(
        fouls, shots, precision, set_piece_r, goals, goals_c,
    )

    return {
        "posesion":        round(poss_avg, 1) if poss_avg is not None else None,
        "tiros":           round(shots,    1),
        "tiros_a_puerta":  round(shots_ot, 1),
        "corners":         round(corners,  1),
        "goles":           round(goals,    2),
        "goles_conc":      round(goals_c,  2),
        "fouls":           round(fouls,    1),
        "amarillas":       round(yellows,  2),
        "rojas":           round(a["weighted_reds"] / w, 3),
        "faltas_prov":     round(suffered, 1),
        "precision":       round(precision,      3),
        "eficiencia":      round(eficiencia,     3),
        "set_piece_ratio": round(set_piece_r,    3),
        "cards_per_foul":  round(cards_per_foul, 3),
        "ratio_fisico":    round(ratio_fisico,   3),
        "tempo":           round(tempo,          1),
        "estilo":          estilo,
        "estilo_desc":     estilo_desc,
        "n_partidos":      a["n_matches"],
    }


# ---------------------------------------------------------------------------
# Normalizacion de dimensiones
# ---------------------------------------------------------------------------

# (clave_output, campo_fuente): todos higher-is-more
_NORM_DIMS: list[tuple[str, str]] = [
    ("posesion",    "posesion"),
    ("tiros",       "tiros"),
    ("precision",   "precision"),
    ("corners",     "corners"),
    ("goles",       "goles"),
    ("eficiencia",  "eficiencia"),
    ("fisico",      "fouls"),
    ("riesgo_tarj", "cards_per_foul"),
    ("faltas_prov", "faltas_prov"),
]


def _normalizar_dims(raw: dict) -> None:
    """Normaliza dimensiones de estilo a escala 1-10 relativa a la liga."""
    for dim_key, campo in _NORM_DIMS:
        valores = [raw[n][campo] for n in raw if raw[n].get(campo) is not None]
        if not valores:
            continue
        min_v, max_v = min(valores), max(valores)
        rng = max_v - min_v if max_v != min_v else 1.0
        for profile in raw.values():
            v = profile.get(campo)
            profile.setdefault("dim_norm", {})[dim_key] = (
                round(1 + ((v - min_v) / rng) * 9, 1) if v is not None else None
            )
