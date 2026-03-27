"""IAP — Índice de Agresividad Ponderado.

  IAP_raw  = faltas × W_F + amarillas × W_A + rojas × W_R
  peso     = e^(-λ × días)
  IAP_team = Σ(IAP_raw × peso) / Σ(peso)
  Escala normalizada 1-10 relativa a la liga.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from core.config import DECAY_LAMBDA, PESO_AMARILLAS, PESO_FALTAS, PESO_ROJAS
from core.helpers import decay_weight, parse_date
from core.utils import TEAM_ALIASES, fuzzy_name_search


def _iap_raw(fouls: int, yellows: int, reds: int) -> float:
    return fouls * PESO_FALTAS + yellows * PESO_AMARILLAS + reds * PESO_ROJAS


def calcular_scores(partidos: list[dict]) -> dict:
    """Calcula scores de agresividad para todos los equipos.

    Devuelve dict[equipo] con general/local/visitante + stats_raw.
    """
    hoy = date.today()
    acum: dict[str, dict] = {}

    for partido in partidos:
        fecha = parse_date(partido["date"])
        peso = decay_weight(fecha, hoy, DECAY_LAMBDA)

        for rol in ("home", "away"):
            team = partido[rol]
            nombre = team["name"]
            iap = _iap_raw(team["fouls"], team["yellow_cards"], team["red_cards"])

            if nombre not in acum:
                acum[nombre] = {
                    "general_num": 0.0, "general_den": 0.0,
                    "local_num":   0.0, "local_den":   0.0,
                    "visit_num":   0.0, "visit_den":   0.0,
                    "n":           0,   "n_local":      0, "n_visit": 0,
                    "faltas_sum":  0,   "amarillas_sum": 0, "rojas_sum": 0,
                }

            a = acum[nombre]
            a["general_num"] += iap * peso
            a["general_den"] += peso
            a["n"]           += 1
            a["faltas_sum"]  += team["fouls"]
            a["amarillas_sum"] += team["yellow_cards"]
            a["rojas_sum"]   += team["red_cards"]

            if rol == "home":
                a["local_num"] += iap * peso
                a["local_den"] += peso
                a["n_local"]   += 1
            else:
                a["visit_num"] += iap * peso
                a["visit_den"] += peso
                a["n_visit"]   += 1

    scores: dict = {}
    for nombre, a in acum.items():
        n = a["n"]
        scores[nombre] = {
            "general":    a["general_num"] / a["general_den"] if a["general_den"] > 0 else 0,
            "local":      a["local_num"]   / a["local_den"]   if a["local_den"]   > 0 else 0,
            "visitante":  a["visit_num"]   / a["visit_den"]   if a["visit_den"]   > 0 else 0,
            "n_partidos": n,
            "n_local":    a["n_local"],
            "n_visitante": a["n_visit"],
            "stats_raw": {
                "faltas_media":    round(a["faltas_sum"]    / n, 2) if n > 0 else 0,
                "amarillas_media": round(a["amarillas_sum"] / n, 2) if n > 0 else 0,
                "rojas_media":     round(a["rojas_sum"]     / n, 2) if n > 0 else 0,
            },
        }

    _normalizar(scores)
    return scores


def _normalizar(scores: dict) -> None:
    """Normaliza a escala 1-10 con min-max relativo a la liga."""
    for dim in ("general", "local", "visitante"):
        valores = [s[dim] for s in scores.values()]
        min_v, max_v = min(valores), max(valores)
        rango = max_v - min_v if max_v != min_v else 1.0
        for s in scores.values():
            s[f"{dim}_norm"] = round(1 + ((s[dim] - min_v) / rango) * 9, 1)


def calcular_rankings(scores: dict) -> dict:
    """Posición de cada equipo en el ranking por dimensión."""
    rankings = {nombre: {} for nombre in scores}
    for dim in ("general", "local", "visitante"):
        orden = sorted(scores, key=lambda n: scores[n][f"{dim}_norm"], reverse=True)
        for pos, nombre in enumerate(orden, 1):
            rankings[nombre][f"rank_{dim}"] = pos
    return rankings


def buscar_equipo(nombre_input: str, scores: dict) -> Optional[str]:
    """Búsqueda del nombre de equipo con fallback progresivo."""
    return fuzzy_name_search(nombre_input, list(scores.keys()), TEAM_ALIASES)
