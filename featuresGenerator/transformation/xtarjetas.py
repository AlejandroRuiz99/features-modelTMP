"""xTarjetas — Tarjetas esperadas y agresividad por volumen."""

from __future__ import annotations

from typing import Optional

from core.config import (
    AGG_VOL_PESO_FALTAS,
    AGG_VOL_PESO_AMARILLAS,
    AGG_VOL_PESO_ROJAS,
)
from core.helpers import safe


def calcular_xtarjetas(
    xfouls_result: dict,
    xstyles: dict,
    equipo_local: str,
    equipo_visitante: str,
    ref_perfiles: Optional[dict] = None,
    arbitro: Optional[str] = None,
) -> dict:
    """Tarjetas esperadas = xFouls × tasa_tarjeta_por_falta × factor_arbitro."""
    sl = xstyles.get(equipo_local,     {})
    sv = xstyles.get(equipo_visitante, {})

    cpf_local     = safe(sl.get("cards_per_foul"), 0.18)
    cpf_visitante = safe(sv.get("cards_per_foul"), 0.18)

    ref_factor = 1.0
    if ref_perfiles and arbitro and arbitro in ref_perfiles:
        ref_factor = safe(ref_perfiles[arbitro].get("factor_amarillas"), 1.0)

    xa_local     = xfouls_result["xfouls_local"]     * cpf_local     * ref_factor
    xa_visitante = xfouls_result["xfouls_visitante"]  * cpf_visitante * ref_factor
    xa_total     = xa_local + xa_visitante

    return {
        "xtarjetas_local":      round(xa_local,     2),
        "xtarjetas_visitante":  round(xa_visitante, 2),
        "xtarjetas_total":      round(xa_total,     2),
        "xamarillas_local":     round(xa_local,     2),
        "xamarillas_visitante": round(xa_visitante, 2),
        "xamarillas_total":     round(xa_total,     2),
        "xrojas_total":         round(safe(sl.get("rojas"), 0.04) + safe(sv.get("rojas"), 0.04), 3),
        "ref_factor_tarjetas":  round(ref_factor,   3),
        "cpf_local":            round(cpf_local,    3),
        "cpf_visitante":        round(cpf_visitante, 3),
    }


def calcular_agresividad_volumen(xfouls_result: dict, xtarjetas: dict) -> dict:
    """Agresividad por volumen: base xFouls + friccion de tarjetas."""
    xf_l = safe(xfouls_result.get("xfouls_local"),     0.0)
    xf_v = safe(xfouls_result.get("xfouls_visitante"), 0.0)
    xf_t = xf_l + xf_v

    ya_l = safe(xtarjetas.get("xamarillas_local"),     safe(xtarjetas.get("xtarjetas_local"),     0.0))
    ya_v = safe(xtarjetas.get("xamarillas_visitante"), safe(xtarjetas.get("xtarjetas_visitante"), 0.0))
    rr_t = safe(xtarjetas.get("xrojas_total"), 0.0)

    share_l = xf_l / xf_t if xf_t > 0 else 0.5
    rr_l = rr_t * share_l
    rr_v = rr_t - rr_l

    vol_l = AGG_VOL_PESO_FALTAS * xf_l + AGG_VOL_PESO_AMARILLAS * ya_l + AGG_VOL_PESO_ROJAS * rr_l
    vol_v = AGG_VOL_PESO_FALTAS * xf_v + AGG_VOL_PESO_AMARILLAS * ya_v + AGG_VOL_PESO_ROJAS * rr_v

    return {
        "local":       round(vol_l, 2),
        "visitante":   round(vol_v, 2),
        "total":       round(vol_l + vol_v, 2),
        "pesos":       {"faltas": AGG_VOL_PESO_FALTAS, "amarillas": AGG_VOL_PESO_AMARILLAS, "rojas": AGG_VOL_PESO_ROJAS},
        "componentes": {
            "xfouls_local":      round(xf_l, 2), "xfouls_visitante":  round(xf_v, 2),
            "xamarillas_local":  round(ya_l, 2), "xamarillas_visitante": round(ya_v, 2),
            "xrojas_local":      round(rr_l, 3), "xrojas_visitante":  round(rr_v, 3),
        },
    }
