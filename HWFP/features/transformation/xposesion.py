"""xPosesion — Posesion de balon esperada por equipo."""

from __future__ import annotations

from HWFP.features.core.helpers import clip, safe


def calcular_xposesion(xstyles: dict, equipo_local: str, equipo_visitante: str) -> dict:
    """Posesion esperada: historico real o proxy por tempo."""
    sl = xstyles.get(equipo_local,     {})
    sv = xstyles.get(equipo_visitante, {})

    poss_l = sl.get("posesion")
    poss_v = sv.get("posesion")

    if poss_l is not None and poss_v is not None:
        poss_local = (poss_l + (100.0 - poss_v)) / 2.0
        fuente = "historico_posesion"
    else:
        tempo_l = safe(sl.get("tempo"), 0)
        tempo_v = safe(sv.get("tempo"), 0)
        total   = tempo_l + tempo_v
        poss_local = clip(100.0 * tempo_l / total, 35.0, 65.0) if total > 0 else 50.0
        fuente = "proxy_tempo"

    poss_local = min(70.0, poss_local + 0.6)  # ligera ventaja local calibrada
    return {
        "posesion_local":     round(poss_local,          1),
        "posesion_visitante": round(100.0 - poss_local,  1),
        "fuente":             fuente,
    }
