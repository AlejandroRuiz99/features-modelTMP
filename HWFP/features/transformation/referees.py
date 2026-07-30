"""Perfilado estadístico de árbitros."""

from __future__ import annotations

from collections import defaultdict

from HWFP.features.core.helpers import safe
from HWFP.features.core.utils import fuzzy_name_search

# Umbral de faltas/partido por equipo para clasificar como equipo "limpio".
# Equipos con promedio histórico < CLEAN_THR son equipos que cometen pocas faltas
# (ej: Barcelona ~9.6, Real Madrid ~9.4). Cuando uno de los dos equipos es limpio,
# el árbitro tiende a pitar menos incluso siendo estricto.
_CLEAN_THR = 11.5
# Mínimo de partidos en cada categoría para que el promedio contextual sea fiable.
_MIN_CONTEXT_SAMPLES = 4


def calcular_perfiles(partidos: list[dict]) -> dict:
    """Estadísticas históricas de cada árbitro.

    Devuelve dict[nombre] con faltas/partido, amarillas/partido,
    factor vs media de la liga, clasificación (estricto/permisivo) y
    promedios contextuales segmentados por tipo de partido:
      - fouls_clean_avg: promedio cuando ≥1 equipo es "limpio" (< 11.5 faltas/partido)
      - fouls_heavy_avg: promedio cuando ambos equipos son "físicos"
    """
    # --- Pasada 0: promedio de faltas cometidas por equipo (para clasificar limpios) ---
    team_acum: dict[str, list[float]] = defaultdict(list)
    for p in partidos:
        fh = safe(p["home"].get("fouls"))
        fa = safe(p["away"].get("fouls"))
        ht = p["home"].get("name", "")
        at = p["away"].get("name", "")
        if fh > 0 and ht:
            team_acum[ht].append(fh)
        if fa > 0 and at:
            team_acum[at].append(fa)

    team_avg: dict[str, float] = {
        t: sum(v) / len(v) for t, v in team_acum.items() if v
    }

    # --- Pasada 1: acumulados por árbitro + segmentación clean/heavy ---
    acum: dict[str, dict] = {}
    ref_clean: dict[str, list[float]] = defaultdict(list)
    ref_heavy: dict[str, list[float]] = defaultdict(list)
    total_fouls = total_yellows = total_matches = 0

    for p in partidos:
        ref = p.get("referee", "")
        if not ref:
            continue

        f = safe(p["home"].get("fouls")) + safe(p["away"].get("fouls"))
        a = safe(p["home"].get("yellow_cards")) + safe(p["away"].get("yellow_cards"))
        r = safe(p["home"].get("red_cards")) + safe(p["away"].get("red_cards"))

        if ref not in acum:
            acum[ref] = {"f": 0, "a": 0, "r": 0, "n": 0}
        acum[ref]["f"] += f
        acum[ref]["a"] += a
        acum[ref]["r"] += r
        acum[ref]["n"] += 1

        total_fouls += f
        total_yellows += a
        total_matches += 1

        if f > 0:
            ht = p["home"].get("name", "")
            at = p["away"].get("name", "")
            h_avg = team_avg.get(ht, _CLEAN_THR + 1)
            a_avg = team_avg.get(at, _CLEAN_THR + 1)
            if h_avg < _CLEAN_THR or a_avg < _CLEAN_THR:
                ref_clean[ref].append(f)
            else:
                ref_heavy[ref].append(f)

    if total_matches == 0:
        return {}

    avg_f = total_fouls / total_matches
    avg_a = total_yellows / total_matches

    _TIPO: list[tuple[float, str]] = [
        (1.20, "muy estricto"),
        (1.05, "estricto"),
        (0.95, "normal"),
        (0.00, "permisivo"),
    ]

    perfiles: dict = {}
    for ref, d in acum.items():
        n = d["n"]
        fp = d["f"] / n
        ap = d["a"] / n
        factor_f = fp / avg_f if avg_f > 0 else 1.0
        factor_a = ap / avg_a if avg_a > 0 else 1.0
        tipo = next(t for threshold, t in _TIPO if factor_a >= threshold)

        clean_list = ref_clean[ref]
        heavy_list = ref_heavy[ref]

        perfiles[ref] = {
            "partidos": n,
            "fouls_partido": round(fp, 1),
            "amarillas_partido": round(ap, 2),
            "rojas_partido": round(d["r"] / n, 2),
            "factor_fouls": round(factor_f, 3),
            "factor_amarillas": round(factor_a, 3),
            "tipo": tipo,
            "avg_liga_fouls": round(avg_f, 1),
            "avg_liga_amarillas": round(avg_a, 2),
            "fouls_clean_avg": (
                round(sum(clean_list) / len(clean_list), 1)
                if len(clean_list) >= _MIN_CONTEXT_SAMPLES
                else None
            ),
            "fouls_heavy_avg": (
                round(sum(heavy_list) / len(heavy_list), 1)
                if len(heavy_list) >= _MIN_CONTEXT_SAMPLES
                else None
            ),
        }

    return perfiles


def buscar_arbitro(nombre_input: str, perfiles: dict) -> str | None:
    """Búsqueda del nombre del árbitro con fallback progresivo."""
    return fuzzy_name_search(nombre_input, list(perfiles.keys()))
