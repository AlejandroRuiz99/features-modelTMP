"""
Calibración automática de agresividad por volumen.

Objetivo:
- Ajustar decay_lambda y alpha_card_pressure (xFouls) para minimizar error en faltas reales.
- Ajustar peso_amarillas y peso_rojas en agresividad_volumen.
"""

from __future__ import annotations

from transformation.xfouls import calcular_xfouls
from transformation.xstyle import calcular_xstyle
from transformation.xtarjetas import calcular_xtarjetas
from core.helpers import safe
from evaluation.base import mae as _mae


def _iter_grid(start: float, stop: float, step: float) -> list[float]:
    vals: list[float] = []
    v = start
    # Redondeo para evitar ruido binario.
    while v <= stop + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def _build_walkforward_samples(partidos: list[dict], n_ultimos: int, warmup: int) -> list[tuple[list[dict], dict]]:
    """
    Devuelve pares (contexto_historico, partido_objetivo) para evaluación walk-forward.
    """
    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:] if n_ultimos > 0 else partidos_sorted
    samples: list[tuple[list[dict], dict]] = []

    for target in muestra:
        contexto = [p for p in partidos_sorted if p["date"] < target["date"]]
        if len(contexto) < warmup:
            continue
        samples.append((contexto, target))
    return samples


def calibrar_agresividad_volumen(
    partidos: list[dict],
    *,
    n_ultimos: int = 240,
    warmup: int = 120,
    decay_lambda_min: float = 0.001,
    decay_lambda_max: float = 0.006,
    decay_lambda_step: float = 0.001,
    alpha_min: float = 0.25,
    alpha_max: float = 0.80,
    alpha_step: float = 0.05,
    peso_amarillas_min: float = 0.10,
    peso_amarillas_max: float = 0.70,
    peso_amarillas_step: float = 0.05,
    peso_rojas_min: float = 0.00,
    peso_rojas_max: float = 1.50,
    peso_rojas_step: float = 0.10,
    arbitro_real: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Calibra alpha y pesos de agresividad por volumen con datos históricos.

    Métrica objetivo:
      minimizar MAE entre predicción y faltas reales totales.
    """
    samples = _build_walkforward_samples(partidos, n_ultimos=n_ultimos, warmup=warmup)
    if not samples:
        return {
            "ok": False,
            "reason": "No hay suficientes partidos para calibrar (muestra vacía).",
            "n_samples": 0,
        }

    # xStyle global para aproximar tasa tarjeta/falta esperada por equipo.
    xstyles = calcular_xstyle(partidos)

    # 1) Calibrar decay_lambda + alpha_card_pressure (xFouls puro)
    best_decay = None
    best_alpha = None
    best_xf_mae = float("inf")
    xf_results: list[dict] = []

    for decay_lambda in _iter_grid(decay_lambda_min, decay_lambda_max, decay_lambda_step):
        for alpha in _iter_grid(alpha_min, alpha_max, alpha_step):
            pred_totals: list[float] = []
            real_totals: list[float] = []
            meta_rows: list[dict] = []

            for contexto, target in samples:
                home = target["home"]["name"]
                away = target["away"]["name"]
                arb = target.get("referee") if arbitro_real else None

                xf = calcular_xfouls(
                    contexto,
                    home,
                    away,
                    arbitro=arb,
                    alpha_card_pressure=alpha,
                    _decay_lambda_override=decay_lambda,
                )
                real_total = safe(target["home"].get("fouls")) + safe(target["away"].get("fouls"))

                pred_totals.append(float(xf["xfouls_total"]))
                real_totals.append(float(real_total))

                xt = calcular_xtarjetas(
                    xf,
                    xstyles,
                    home,
                    away,
                    ref_perfiles=None,
                    arbitro=arb,
                )
                meta_rows.append(
                    {
                        "xf_total": float(xf["xfouls_total"]),
                        "xamarillas_total": float(xt.get("xamarillas_total", xt.get("xtarjetas_total", 0.0))),
                        "xrojas_total": float(xt.get("xrojas_total", 0.0)),
                        "real_fouls_total": float(real_total),
                    }
                )

            mae_alpha = _mae(pred_totals, real_totals)
            xf_results.append(
                {
                    "decay_lambda": decay_lambda,
                    "alpha": alpha,
                    "mae_xfouls": round(mae_alpha, 4),
                    "rows": meta_rows,
                }
            )
            if mae_alpha < best_xf_mae:
                best_xf_mae = mae_alpha
                best_decay = decay_lambda
                best_alpha = alpha

    assert best_decay is not None
    assert best_alpha is not None
    best_rows = next(
        r["rows"]
        for r in xf_results
        if r["alpha"] == best_alpha and r["decay_lambda"] == best_decay
    )

    # 2) Calibrar pesos (agresividad_volumen = xF + wY*xA + wR*xR)
    best_wy = 0.35
    best_wr = 0.75
    best_vol_mae = float("inf")

    for wy in _iter_grid(peso_amarillas_min, peso_amarillas_max, peso_amarillas_step):
        for wr in _iter_grid(peso_rojas_min, peso_rojas_max, peso_rojas_step):
            pred: list[float] = []
            real: list[float] = []
            for row in best_rows:
                pred_total = row["xf_total"] + wy * row["xamarillas_total"] + wr * row["xrojas_total"]
                pred.append(pred_total)
                real.append(row["real_fouls_total"])
            mae = _mae(pred, real)
            if mae < best_vol_mae:
                best_vol_mae = mae
                best_wy = wy
                best_wr = wr

    improvement = 0.0
    if best_xf_mae > 0:
        improvement = (best_xf_mae - best_vol_mae) / best_xf_mae

    result = {
        "ok": True,
        "n_samples": len(best_rows),
        "best_decay_lambda": round(best_decay, 4),
        "best_alpha_card_pressure": round(best_alpha, 4),
        "best_peso_amarillas": round(best_wy, 4),
        "best_peso_rojas": round(best_wr, 4),
        "mae_xfouls_base": round(best_xf_mae, 4),
        "mae_agresividad_volumen": round(best_vol_mae, 4),
        "improvement_pct": round(improvement * 100.0, 2),
        "search_space": {
            "decay_lambda": [decay_lambda_min, decay_lambda_max, decay_lambda_step],
            "alpha": [alpha_min, alpha_max, alpha_step],
            "peso_amarillas": [peso_amarillas_min, peso_amarillas_max, peso_amarillas_step],
            "peso_rojas": [peso_rojas_min, peso_rojas_max, peso_rojas_step],
        },
    }

    if verbose:
        print("\n" + "═" * 58)
        print("  CALIBRACIÓN AUTOMÁTICA — AGRESIVIDAD POR VOLUMEN")
        print("─" * 58)
        print(f"  Muestras walk-forward: {result['n_samples']}")
        print(f"  decay_lambda óptimo:        {result['best_decay_lambda']}")
        print(f"  alpha_card_pressure óptimo: {result['best_alpha_card_pressure']}")
        print(f"  peso_amarillas óptimo:      {result['best_peso_amarillas']}")
        print(f"  peso_rojas óptimo:          {result['best_peso_rojas']}")
        print(f"  MAE xFouls base:            {result['mae_xfouls_base']}")
        print(f"  MAE agresividad_volumen:    {result['mae_agresividad_volumen']}")
        print(f"  Mejora relativa:            {result['improvement_pct']}%")
        print("═" * 58 + "\n")

    return result


if __name__ == "__main__":
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli("Calibracion de agresividad por volumen", calibrable=True)
    partidos = load_partidos()
    result = calibrar_agresividad_volumen(partidos, n_ultimos=args.n)
    print(f"\nMejor configuracion: {result}")
