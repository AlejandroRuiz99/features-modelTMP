from __future__ import annotations

from typing import Any


def build_quality_category(contract: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    """
    Evalua la completitud del JSON contrato unificado.

    Verifica la presencia de los campos criticos que predictionModels necesita
    para producir predicciones de calidad.
    """
    missing = []

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

    coverage = max(0.0, 1.0 - len(missing) * 0.15)
    return {
        "coverage_score": round(coverage, 3),
        "missing_blocks": missing,
        "warnings": warnings,
    }
