"""Knowledge Pack — orquestador que ensambla todas las metricas del partido.

El contexto competitivo (ICC, factores) llega PRE-COMPUTADO desde
competitive_context.py; este modulo no recomputa ni infiere nada de el.
"""

from __future__ import annotations


from HWFP.features.core.helpers import clip
from HWFP.features.transformation.xgoals import calcular_xgoals, poisson_outcome_probs
from HWFP.features.transformation.xposesion import calcular_xposesion
from HWFP.features.transformation.xtarjetas import calcular_xtarjetas, calcular_agresividad_volumen
from HWFP.features.transformation.forma import calcular_forma_reciente, calcular_contexto_temporada
from HWFP.features.transformation.match_profile import (
    calcular_compatibilidad_estilos,
    calcular_xvolumen_eventos,
    generar_narrative,
)


# Factores neutros para cuando no hay contexto competitivo disponible.
_NEUTRAL_FACTORS = {"xg_factor": 1.0, "xfouls_factor": 1.0, "posesion_delta_pp": 0.0}
_NEUTRAL_CTX = {
    "local": {
        "factors": _NEUTRAL_FACTORS,
        "scores": {"icc": None},
        "lectura": "sin datos",
    },
    "visitante": {
        "factors": _NEUTRAL_FACTORS,
        "scores": {"icc": None},
        "lectura": "sin datos",
    },
}


def ensamblar_knowledge_pack(
    equipo_local: str,
    equipo_visitante: str,
    partidos: list[dict],
    xstyles: dict,
    xfouls_result: dict,
    ref_perfiles: dict | None = None,
    arbitro: str | None = None,
    jornada: int | None = None,
    contexto_comp: dict | None = None,
) -> dict:
    """Ensambla el knowledge pack completo para el partido.

    Args:
        contexto_comp: dict con factores ICC pre-computados por competitive_context.py.
                       Estructura: {"local": {"factors": {...}, "scores": {"icc": ...},
                       "lectura": ...}, "visitante": {...}}.
                       Si None, se usan factores neutros (sin ajuste de contexto).
    """
    ctx = contexto_comp or _NEUTRAL_CTX

    forma_local = calcular_forma_reciente(partidos, equipo_local)
    forma_visitante = calcular_forma_reciente(partidos, equipo_visitante)
    contexto = calcular_contexto_temporada(partidos, equipo_local, jornada)
    xgoals = calcular_xgoals(xstyles, equipo_local, equipo_visitante)
    xposesion = calcular_xposesion(xstyles, equipo_local, equipo_visitante)
    xtarjetas = calcular_xtarjetas(
        xfouls_result, xstyles, equipo_local, equipo_visitante, ref_perfiles, arbitro
    )
    xvol = calcular_xvolumen_eventos(xstyles, equipo_local, equipo_visitante, ctx)

    # Aplicar factores del contexto competitivo
    f_xg_l = ctx["local"]["factors"]["xg_factor"]
    f_xg_v = ctx["visitante"]["factors"]["xg_factor"]
    xg_l_adj = round(xgoals["xg_local"] * f_xg_l, 2)
    xg_v_adj = round(xgoals["xg_visitante"] * f_xg_v, 2)
    xgoals_adj = {
        "xg_local": xg_l_adj,
        "xg_visitante": xg_v_adj,
        "xg_total": round(xg_l_adj + xg_v_adj, 2),
        **poisson_outcome_probs(xg_l_adj, xg_v_adj),
    }

    f_xf_l = ctx["local"]["factors"]["xfouls_factor"]
    f_xf_v = ctx["visitante"]["factors"]["xfouls_factor"]
    xf_l_adj = round(xfouls_result["xfouls_local"] * f_xf_l, 1)
    xf_v_adj = round(xfouls_result["xfouls_visitante"] * f_xf_v, 1)
    xfouls_adj = {
        "local": xf_l_adj,
        "visitante": xf_v_adj,
        "total": round(xf_l_adj + xf_v_adj, 1),
        "avg_liga": xfouls_result["avg_liga"],
    }

    delta_l = ctx["local"]["factors"]["posesion_delta_pp"]
    delta_v = ctx["visitante"]["factors"]["posesion_delta_pp"]
    poss_l_adj = clip(xposesion["posesion_local"] + delta_l - delta_v, 35.0, 65.0)
    xposesion_adj = {
        "posesion_local": round(poss_l_adj, 1),
        "posesion_visitante": round(100.0 - poss_l_adj, 1),
        "fuente": xposesion["fuente"],
    }

    ratio_l = xf_l_adj / max(0.1, xfouls_result["xfouls_local"])
    ratio_v = xf_v_adj / max(0.1, xfouls_result["xfouls_visitante"])
    xtarjetas_adj = {
        **xtarjetas,
        "xtarjetas_local": round(xtarjetas["xtarjetas_local"] * ratio_l, 2),
        "xtarjetas_visitante": round(xtarjetas["xtarjetas_visitante"] * ratio_v, 2),
        "xtarjetas_total": round(
            xtarjetas["xtarjetas_local"] * ratio_l
            + xtarjetas["xtarjetas_visitante"] * ratio_v,
            2,
        ),
        "xamarillas_local": round(xtarjetas["xamarillas_local"] * ratio_l, 2),
        "xamarillas_visitante": round(xtarjetas["xamarillas_visitante"] * ratio_v, 2),
        "xamarillas_total": round(
            xtarjetas["xamarillas_local"] * ratio_l
            + xtarjetas["xamarillas_visitante"] * ratio_v,
            2,
        ),
    }

    agresividad_vol = calcular_agresividad_volumen(
        {"xfouls_local": xf_l_adj, "xfouls_visitante": xf_v_adj},
        xtarjetas_adj,
    )
    compat = calcular_compatibilidad_estilos(xstyles, equipo_local, equipo_visitante)
    narrative = generar_narrative(
        equipo_local,
        equipo_visitante,
        forma_local,
        forma_visitante,
        xgoals_adj,
        xposesion_adj,
        xtarjetas_adj,
        xfouls_adj,
        agresividad_vol,
        ctx,
        compat,
        contexto,
    )

    return {
        "forma": {"local": forma_local, "visitante": forma_visitante},
        "contexto_temporada": contexto,
        "contexto_competitivo": ctx,
        "expected_metrics": {
            "xgoals": xgoals_adj,
            "xposesion": xposesion_adj,
            "xtarjetas": xtarjetas_adj,
            "agresividad_volumen": agresividad_vol,
            "xfouls": xfouls_adj,
            "xvolumen_eventos": xvol,
        },
        "base_metrics": {
            "xgoals": xgoals,
            "xposesion": xposesion,
            "xtarjetas": xtarjetas,
            "xfouls": {
                "local": xfouls_result["xfouls_local"],
                "visitante": xfouls_result["xfouls_visitante"],
                "total": xfouls_result["xfouls_total"],
                "avg_liga": xfouls_result["avg_liga"],
            },
        },
        "compatibilidad_estilos": compat,
        "narrative": narrative,
        "market_signal": None,
    }
