"""
xFouls — Predicción de faltas esperadas en un partido concreto.

  xFouls_A = base_A × draw_factor_B × ref_factor × card_pressure_A

  base_A          tasa histórica de faltas (con decay temporal)
  draw_factor_B   cuánto provoca el rival (fouls drawn / avg)
  ref_factor      leniencia del árbitro vs media liga
  card_pressure   autorregulación cuando el ratio tarjetas/falta es alto
"""

from __future__ import annotations

from datetime import date


from HWFP.features.core.config import ALPHA_CARD_PRESSURE, DECAY_LAMBDA
from HWFP.features.core.helpers import decay_weight, parse_date, safe
from HWFP.features.transformation.referees import calcular_perfiles

_RATIO_AMARILLAS_FALTA_FALLBACK = 2.3 / 12.6  # usado solo si no hay datos


def calcular_xfouls(
    partidos: list[dict],
    equipo_local: str,
    equipo_visitante: str,
    arbitro: str | None = None,
    alpha_card_pressure: float | None = None,
    _decay_lambda_override: float | None = None,
) -> dict:
    """
    Calcula las faltas esperadas en un partido concreto.

    _decay_lambda_override: si se indica, sobreescribe DECAY_LAMBDA del config
      (útil para calibración sin modificar config.yaml).
    """
    lam = _decay_lambda_override if _decay_lambda_override is not None else DECAY_LAMBDA
    hoy = date.today()
    t: dict = {}
    total_fouls = total_yellows = total_matches = 0

    for p in partidos:
        fecha = parse_date(p["date"])
        peso = decay_weight(fecha, hoy, lam)

        for rol, opp_rol in [("home", "away"), ("away", "home")]:
            team = p[rol]
            nombre = team["name"]
            opp = p[opp_rol]

            f_com = safe(team.get("fouls"))
            f_drawn = safe(opp.get("fouls"))
            yellows = safe(team.get("yellow_cards"))

            if nombre not in t:
                t[nombre] = {
                    "f_num": 0.0,
                    "f_den": 0.0,
                    "d_num": 0.0,
                    "d_den": 0.0,
                    "y_sum": 0.0,
                    "n": 0,
                }

            t[nombre]["f_num"] += f_com * peso
            t[nombre]["f_den"] += peso
            t[nombre]["d_num"] += f_drawn * peso
            t[nombre]["d_den"] += peso
            t[nombre]["y_sum"] += yellows
            t[nombre]["n"] += 1

        total_fouls += safe(p["home"].get("fouls")) + safe(p["away"].get("fouls"))
        total_yellows += safe(p["home"].get("yellow_cards")) + safe(p["away"].get("yellow_cards"))
        total_matches += 1

    avg_total = total_fouls / total_matches if total_matches else 25.2
    avg_equipo = avg_total / 2
    avg_yellows_equipo = (total_yellows / total_matches / 2) if total_matches else 2.3
    ratio_amarillas_falta_avg = (
        avg_yellows_equipo / avg_equipo if avg_equipo > 0 else _RATIO_AMARILLAS_FALTA_FALLBACK
    )

    def fouls_rate(nombre: str) -> float:
        d = t.get(nombre, {})
        den = d.get("f_den", 0)
        return d["f_num"] / den if den > 0 else avg_equipo

    def drawn_rate(nombre: str) -> float:
        d = t.get(nombre, {})
        den = d.get("d_den", 0)
        return d["d_num"] / den if den > 0 else avg_equipo

    alpha_cp = (
        ALPHA_CARD_PRESSURE if alpha_card_pressure is None else alpha_card_pressure
    )

    def card_pressure(nombre: str) -> float:
        d = t.get(nombre, {})
        n = d.get("n", 0)
        if n == 0:
            return 1.0
        yr = d["y_sum"] / n
        fr = fouls_rate(nombre)
        ratio = yr / fr if fr > 0 else ratio_amarillas_falta_avg
        exceso = max(0.0, ratio - ratio_amarillas_falta_avg)
        return max(0.80, 1.0 - alpha_cp * exceso)

    ref_factor = 1.0
    arbitro_info: dict | None = None

    perfiles_refs = calcular_perfiles(partidos)
    if arbitro and arbitro in perfiles_refs:
        ref = perfiles_refs[arbitro]
        ref_factor = ref["factor_fouls"]
        arbitro_info = {"nombre": arbitro, **ref}

    base_loc = fouls_rate(equipo_local)
    draw_vis = drawn_rate(equipo_visitante)
    draw_factor_vis = draw_vis / avg_equipo
    cp_loc = card_pressure(equipo_local)
    xf_local = base_loc * draw_factor_vis * ref_factor * cp_loc

    base_vis = fouls_rate(equipo_visitante)
    draw_loc = drawn_rate(equipo_local)
    draw_factor_loc = draw_loc / avg_equipo
    cp_vis = card_pressure(equipo_visitante)
    xf_vis = base_vis * draw_factor_loc * ref_factor * cp_vis

    return {
        "xfouls_local": round(xf_local, 1),
        "xfouls_visitante": round(xf_vis, 1),
        "xfouls_total": round(xf_local + xf_vis, 1),
        "avg_liga": round(avg_total, 1),
        "base_local": round(base_loc, 1),
        "base_visitante": round(base_vis, 1),
        "draw_factor_local": round(draw_factor_loc, 2),
        "draw_factor_visitante": round(draw_factor_vis, 2),
        "card_pressure_local": round(cp_loc, 2),
        "card_pressure_visitante": round(cp_vis, 2),
        "ref_factor": round(ref_factor, 2),
        "arbitro": arbitro_info,
        "fouls_drawn_local": round(draw_loc, 1),
        "fouls_drawn_visitante": round(draw_vis, 1),
    }
