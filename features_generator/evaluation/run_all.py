"""
run_all.py — Orquestador de todas las evaluaciones y calibraciones.

Ejecuta en orden:
  1. xFouls      — validación + calibración opcional de DECAY_LAMBDA y ALPHA_CARD_PRESSURE
  2. Árbitros    — valor añadido del factor árbitro
  3. IAP         — validación/calibración de pesos (faltas, amarillas)
  4. xGoals      — backtesting xG vs goles reales
  5. xPosesión   — backtesting posesión vs dato real
  6. Agresividad — calibración opcional agresividad por volumen (--calibrar)

Con --update-config actualiza config.yaml solo si los parámetros calibrados
mejoran el MAE/correlación respecto a los valores actuales.

Uso:
  python -m evaluate.run_all                   # solo validación (read-only)
  python -m evaluate.run_all --calibrar        # calibración + validación
  python -m evaluate.run_all --update-config   # calibrar y actualizar config.yaml
  python -m evaluate.run_all --n 200           # usar 200 partidos recientes
"""

from __future__ import annotations

from evaluation import (
    xfouls_eval,
    referee_eval,
    iap_eval,
    xgoals_eval,
    xposesion_eval,
    aggressivity_eval,
)
from evaluation.base import header, footer


# ---------------------------------------------------------------------------
# Función principal (acepta datos pre-cargados para evitar doble fetch)
# ---------------------------------------------------------------------------


def evaluar_todos(
    partidos: list,
    xstyles: dict,
    n_ultimos: int = 150,
    calibrar: bool = False,
    update_config: bool = False,
) -> dict:
    """Ejecuta todas las evaluaciones con datos ya cargados."""
    do_calibrar = calibrar or update_config
    summary: dict = {}

    header("[1/6] xFouls")
    xf = xfouls_eval.evaluar(partidos, n_ultimos=n_ultimos)
    summary["xfouls"] = xf
    if do_calibrar:
        cal_xf = xfouls_eval.calibrar(partidos, n_ultimos=n_ultimos)
        summary["xfouls_calibracion"] = cal_xf
        if update_config:
            mae_actual = (xf.get("total") or {}).get("mae", float("inf"))
            if cal_xf["best_mae"] < mae_actual:
                xfouls_eval._update_config(
                    cal_xf["best_decay_lambda"], cal_xf["best_alpha_card_pressure"]
                )
                print(
                    f"  [OK] config.yaml → λ={cal_xf['best_decay_lambda']}  α={cal_xf['best_alpha_card_pressure']}"
                )
            else:
                print("  [INFO] xFouls: parámetros actuales ya son óptimos.")

    header("[2/6] Factor Árbitro")
    ref = referee_eval.evaluar(partidos, n_ultimos=n_ultimos)
    summary["referee"] = ref

    header("[3/6] IAP — Índice de Agresividad Ponderado")
    iap = iap_eval.evaluar(partidos, n_ultimos=n_ultimos)
    summary["iap"] = iap
    if do_calibrar:
        cal_iap = iap_eval.calibrar(partidos, n_ultimos=n_ultimos)
        summary["iap_calibracion"] = cal_iap
        if update_config:
            r_actual = iap.get("correlacion_r", -1)
            if cal_iap["best_r"] > r_actual:
                iap_eval._update_config(
                    cal_iap["best_peso_faltas"], cal_iap["best_peso_amarillas"]
                )
                print(
                    f"  [OK] config.yaml → peso_faltas={cal_iap['best_peso_faltas']}  peso_amarillas={cal_iap['best_peso_amarillas']}"
                )
            else:
                print("  [INFO] IAP: pesos actuales ya son óptimos.")

    header("[4/6] xGoals")
    xg = xgoals_eval.evaluar(partidos, xstyles, n_ultimos=n_ultimos)
    summary["xgoals"] = xg

    header("[5/6] xPosesión")
    xp = xposesion_eval.evaluar(partidos, xstyles, n_ultimos=n_ultimos)
    summary["xposesion"] = xp

    header("[6/6] Agresividad volumen")
    if do_calibrar:
        cal_agg = aggressivity_eval.calibrar_agresividad_volumen(
            partidos, n_ultimos=n_ultimos
        )
        summary["aggressivity_calibracion"] = cal_agg
    else:
        print("  (solo se ejecuta con --calibrar)")

    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI (fetch propio, para uso standalone)
# ---------------------------------------------------------------------------


def main():
    from transformation import calcular_xstyle
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli(
        "Validación y calibración completa del features_generator", calibrable=True
    )
    partidos = load_partidos()

    evaluar_todos(
        partidos,
        calcular_xstyle(partidos),
        n_ultimos=args.n,
        calibrar=args.calibrar,
        update_config=args.update_config,
    )


# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------


def _print_summary(s: dict) -> None:
    header("RESUMEN — VALIDACIÓN DEL FEATURES_GENERATOR")

    xf = s.get("xfouls", {}).get("total", {})
    if xf:
        mae_v = xf.get("mae", float("nan"))
        r_v = xf.get("r", float("nan"))
        semaforo = "✓" if mae_v < 4.5 else ("⚠" if mae_v < 6.0 else "✗")
        print(f"  xFouls       MAE={mae_v:.3f}  r={r_v:.3f}  {semaforo}")

    ref = s.get("referee", {})
    if ref:
        mejora = ref.get("mejora_mae", 0)
        semaforo = "✓" if mejora > 0.1 else ("~" if mejora > -0.1 else "✗")
        print(f"  Árbitro      mejora_MAE={mejora:+.3f}  {semaforo}")

    iap = s.get("iap", {})
    if iap:
        r_iap = iap.get("correlacion_r", float("nan"))
        semaforo = "✓" if r_iap > 0.30 else ("⚠" if r_iap > 0.15 else "✗")
        print(f"  IAP          r={r_iap:.3f}  {semaforo}")

    xg = s.get("xgoals", {}).get("over25", {})
    if xg:
        bss = xg.get("bss", float("nan"))
        semaforo = "✓" if bss > 0.05 else ("⚠" if bss > 0 else "✗")
        print(f"  xGoals OU2.5 BSS={bss:+.4f}  {semaforo}")

    xp = s.get("xposesion", {})
    if xp:
        mae_p = xp.get("mae_pp", float("nan"))
        semaforo = "✓" if mae_p < 5 else ("⚠" if mae_p < 10 else "✗")
        print(f"  xPosesión    MAE={mae_p:.2f}pp  {semaforo}")

    print(f"\n  Leyenda: ✓=bueno  ⚠=mejorable  ✗=sin valor predictivo")
    footer()


if __name__ == "__main__":
    main()
