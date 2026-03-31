"""Forma reciente y contexto de temporada."""

from __future__ import annotations


from core.config import FORMA_VENTANA, JORNADAS_LALIGA
from core.helpers import safe

_PTS = {"W": 3, "D": 1, "L": 0}


def _resultado_para_equipo(partido: dict, equipo: str) -> str:
    es_local = partido["home"]["name"] == equipo
    gf = safe((partido["home"] if es_local else partido["away"]).get("goals"))
    gc = safe((partido["away"] if es_local else partido["home"]).get("goals"))
    return "W" if gf > gc else "D" if gf == gc else "L"


def calcular_forma_reciente(
    partidos: list[dict],
    equipo: str,
    n: int = FORMA_VENTANA,
) -> dict:
    """Analiza los ultimos N partidos del equipo.

    Retorna: puntos, goles, faltas, tarjetas, racha y tendencia.
    """
    ultimos = sorted(
        [
            p
            for p in partidos
            if p["home"]["name"] == equipo or p["away"]["name"] == equipo
        ],
        key=lambda p: p["date"],
        reverse=True,
    )[:n]

    if not ultimos:
        return {
            "partidos_analizados": 0,
            "puntos": 0,
            "puntos_media": 0.0,
            "victorias": 0,
            "empates": 0,
            "derrotas": 0,
            "goles_anotados_media": 0.0,
            "goles_recibidos_media": 0.0,
            "faltas_media": 0.0,
            "tarjetas_media": 0.0,
            "racha": [],
            "racha_str": "—",
            "tendencia": "sin_datos",
        }

    puntos = goles_a = goles_c = faltas = tarjetas = 0
    victorias = empates = derrotas = 0
    racha: list[str] = []

    for p in ultimos:
        es_local = p["home"]["name"] == equipo
        team = p["home" if es_local else "away"]
        opp = p["away" if es_local else "home"]

        res = _resultado_para_equipo(p, equipo)
        racha.append(res)
        puntos += _PTS[res]
        if res == "W":
            victorias += 1
        elif res == "D":
            empates += 1
        else:
            derrotas += 1

        goles_a += safe(team.get("goals"))
        goles_c += safe(opp.get("goals"))
        faltas += safe(team.get("fouls"))
        tarjetas += safe(team.get("yellow_cards")) + safe(team.get("red_cards")) * 2

    n_real = len(ultimos)
    tendencia = "estable"
    if n_real >= 4:
        mid = n_real // 2
        pts_rec = sum(_PTS[r] for r in racha[:mid])
        pts_ant = sum(_PTS[r] for r in racha[mid:])
        max_pts = mid * 3
        if max_pts > 0:
            diff = (pts_rec - pts_ant) / max_pts
            if diff > 0.20:
                tendencia = "mejorando"
            elif diff < -0.20:
                tendencia = "empeorando"

    return {
        "partidos_analizados": n_real,
        "puntos": puntos,
        "puntos_media": round(puntos / n_real, 2),
        "victorias": victorias,
        "empates": empates,
        "derrotas": derrotas,
        "goles_anotados_media": round(goles_a / n_real, 2),
        "goles_recibidos_media": round(goles_c / n_real, 2),
        "faltas_media": round(faltas / n_real, 1),
        "tarjetas_media": round(tarjetas / n_real, 2),
        "racha": racha,
        "racha_str": "".join(racha),
        "tendencia": tendencia,
    }


def calcular_contexto_temporada(
    partidos: list[dict],
    equipo_local: str,
    jornada: int | None = None,
) -> dict:
    """Determina el tramo de temporada y la presion de cierre."""
    if jornada is None:
        season = max((p.get("season", 0) for p in partidos), default=2025)
        jornada = (
            sum(
                1
                for p in partidos
                if p.get("season") == season
                and (
                    p["home"]["name"] == equipo_local
                    or p["away"]["name"] == equipo_local
                )
            )
            + 1
        )

    jornadas_restantes = max(0, JORNADAS_LALIGA - jornada)

    if jornada <= 10:
        tramo, tramo_desc, presion = "inicio", "inicio de temporada (rodaje)", "baja"
    elif jornada <= 26:
        tramo, tramo_desc, presion = "medio", "tramo central de la temporada", "media"
    else:
        tramo, tramo_desc, presion = "final", "recta final de la temporada", "alta"

    return {
        "jornada_estimada": jornada,
        "jornadas_totales": JORNADAS_LALIGA,
        "jornadas_restantes": jornadas_restantes,
        "tramo": tramo,
        "tramo_desc": tramo_desc,
        "presion_final": presion,
        "porcentaje_temporada": round(jornada / JORNADAS_LALIGA * 100, 1),
    }
