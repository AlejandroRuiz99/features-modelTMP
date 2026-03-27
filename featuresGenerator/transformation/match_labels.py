"""Labels categoricos del partido: intensidad fisica, riesgo disciplinario, sesgo mercado."""

from __future__ import annotations

from typing import Any


def build_labels_category(
    kp: dict[str, Any],
    *,
    is_derby: bool = False,
    aggressiveness_total: float = 0.5,
    urgency_home: float | None = None,
    urgency_away: float | None = None,
    fatigue_home: float = 0.2,
    fatigue_away: float = 0.2,
) -> dict[str, Any]:
    """Calcula labels categoricos combinando metricas y contexto.

    La intensidad y el riesgo no dependen solo de xFouls/xTarjetas brutos:
    un derby con equipos agresivos y fatigados tras Champions eleva ambos.
    """
    ms = kp.get("market_signal") or {}
    xt = kp.get("expected_metrics", {}).get("xtarjetas", {})
    xf = kp.get("expected_metrics", {}).get("xfouls", {})
    xt_total = float(xt.get("xtarjetas_total", 0) or 0)
    xf_total = float(xf.get("total", 0) or 0)

    # --- Intensidad esperada ---
    intensity_score = xf_total / 30.0
    if is_derby:
        intensity_score += 0.15
    intensity_score += max(0.0, aggressiveness_total - 0.5) * 0.3
    avg_fatigue = (fatigue_home + fatigue_away) / 2.0
    if avg_fatigue > 0.35:
        intensity_score += (avg_fatigue - 0.35) * 0.2

    if intensity_score >= 0.85:
        intensidad = "alta"
    elif intensity_score >= 0.70:
        intensidad = "media"
    else:
        intensidad = "baja"

    # --- Riesgo disciplinario ---
    risk_score = xt_total / 8.0
    if is_derby:
        risk_score += 0.12
    risk_score += max(0.0, aggressiveness_total - 0.5) * 0.25
    urgencies = [u for u in (urgency_home, urgency_away) if u is not None]
    if urgencies:
        avg_urgency = sum(urgencies) / len(urgencies)
        if avg_urgency > 0.6:
            risk_score += (avg_urgency - 0.6) * 0.15

    if risk_score >= 0.80:
        riesgo = "alto"
    elif risk_score >= 0.55:
        riesgo = "medio"
    else:
        riesgo = "bajo"

    return {
        "tipo_partido": kp.get("compatibilidad_estilos", {}).get("tipo_partido", "PARTIDO EQUILIBRADO"),
        "intensidad_esperada": intensidad,
        "sesgo_mercado_vs_modelo": "alineado" if (ms.get("global_alignment", {}).get("score") or 0) >= 0.6 else "divergente",
        "riesgo_disciplinario": riesgo,
    }
