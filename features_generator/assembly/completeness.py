from __future__ import annotations

from typing import Any

# Penalidades por campo faltante. Campos críticos para el modelo tienen peso mayor.
# La suma total es >1 intencionalmente: si falta todo, coverage_score queda en 0.0.
_FIELD_PENALTIES: dict[str, float] = {
    "arbitro.nombre": 0.15,
    "arbitro.estadisticas (sin historico GMM)": 0.20,
    "equipos.local.temporada_completa.faltas_cometidas": 0.15,
    "equipos.visitante.temporada_completa.faltas_cometidas": 0.15,
    "metricas_esperadas.xfaltas": 0.20,
    "metricas_esperadas.agresividad": 0.05,
    "mercado.faltas_total_ou.linea": 0.05,
    "mercado.resultado": 0.05,
    "equipos.local.clasificacion.posicion": 0.03,
    "equipos.visitante.clasificacion.posicion": 0.03,
}


def build_quality_category(
    contract: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    """
    Evalua la completitud del JSON contrato unificado.

    Verifica la presencia de los campos criticos que prediction_models necesita
    para producir predicciones de calidad. Los campos mas importantes para el
    modelo tienen mayor penalidad que los opcionales (mercado, clasificacion).
    """
    missing: list[str] = []

    # Arbitro
    if not (contract.get("arbitro", {}).get("nombre") or "").strip():
        missing.append("arbitro.nombre")
    arb_stats = contract.get("arbitro", {}).get("estadisticas", {})
    if not arb_stats or arb_stats.get("partidos_arbitrados", 0) == 0:
        missing.append("arbitro.estadisticas (sin historico GMM)")

    # Equipos — campos criticos
    for rol in ("local", "visitante"):
        eq = contract.get("equipos", {}).get(rol, {})
        if not eq.get("temporada_completa", {}).get("faltas_cometidas"):
            missing.append(f"equipos.{rol}.temporada_completa.faltas_cometidas")
        if eq.get("clasificacion", {}).get("posicion") is None:
            missing.append(f"equipos.{rol}.clasificacion.posicion")

    # Metricas esperadas
    em = contract.get("metricas_esperadas", {})
    if not em.get("xfaltas", {}).get("local"):
        missing.append("metricas_esperadas.xfaltas")
    if em.get("agresividad", {}).get("local") is None:
        missing.append("metricas_esperadas.agresividad")

    # Mercado de faltas
    linea = contract.get("mercado", {}).get("faltas_total_ou", {}).get("linea")
    if not linea:
        missing.append("mercado.faltas_total_ou.linea")

    # Probabilidades de mercado 1X2
    prob_local = contract.get("mercado", {}).get("resultado", {}).get("prob_local")
    if prob_local is None:
        missing.append("mercado.resultado")

    penalty = sum(_FIELD_PENALTIES.get(f, 0.05) for f in missing)
    coverage = max(0.0, 1.0 - penalty)
    return {
        "coverage_score": round(coverage, 3),
        "missing_blocks": missing,
        "warnings": warnings,
    }
