"""
xgoals_eval.py — Backtesting de xGoals vs goles reales.

Métricas:
  - MAE, RMSE y bias de xG por equipo
  - Log-loss y Brier score para Over 2.5, BTTS, 1X2
  - Brier Skill Score (BSS): ¿mejor que predecir la frecuencia base?

Uso:
  python -m evaluate.xgoals_eval
"""

from __future__ import annotations

from core.helpers import safe
from transformation.xgoals import calcular_xgoals
from evaluation.base import mae, rmse, bias, log_loss, brier, brier_skill, header, section, row, footer

MIN_CONTEXT = 20


def evaluar(
    partidos: list[dict],
    xstyles: dict,
    n_ultimos: int = 150,
    verbose: bool = True,
) -> dict:
    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:]

    preds_l, reals_l = [], []
    preds_v, reals_v = [], []
    prob_ov25, real_ov25 = [], []
    prob_btts, real_btts = [], []
    prob_1,    real_1    = [], []
    prob_x,    real_x    = [], []
    prob_2,    real_2    = [], []

    for partido in muestra:
        ea = partido["home"]["name"]
        eb = partido["away"]["name"]
        if ea not in xstyles or eb not in xstyles:
            continue
        contexto = [p for p in partidos_sorted if p["date"] < partido["date"]]
        if len(contexto) < MIN_CONTEXT:
            continue

        xg = calcular_xgoals(xstyles, ea, eb)
        g_l = safe(partido["home"].get("goals"))
        g_v = safe(partido["away"].get("goals"))
        g_t = g_l + g_v

        preds_l.append(xg["xg_local"]); reals_l.append(float(g_l))
        preds_v.append(xg["xg_visitante"]); reals_v.append(float(g_v))
        prob_ov25.append(xg["prob_over25"]); real_ov25.append(1 if g_t > 2 else 0)
        prob_btts.append(xg["prob_btts"]); real_btts.append(1 if g_l > 0 and g_v > 0 else 0)
        prob_1.append(xg["prob_local_win"]); real_1.append(1 if g_l > g_v else 0)
        prob_x.append(xg["prob_draw"]); real_x.append(1 if g_l == g_v else 0)
        prob_2.append(xg["prob_visitante_win"]); real_2.append(1 if g_v > g_l else 0)

    n = len(preds_l)
    if n == 0:
        if verbose:
            print("[WARN] Sin datos suficientes para evaluar xGoals.")
        return {}

    resultados = {
        "n": n,
        "xgoals": {
            "mae_local":     round(mae(preds_l, reals_l), 3),
            "rmse_local":    round(rmse(preds_l, reals_l), 3),
            "bias_local":    round(bias(preds_l, reals_l), 3),
            "mae_visitante": round(mae(preds_v, reals_v), 3),
            "rmse_visitante":round(rmse(preds_v, reals_v), 3),
            "bias_visitante":round(bias(preds_v, reals_v), 3),
        },
        "over25": {
            "log_loss": round(log_loss(prob_ov25, real_ov25), 4),
            "brier":    round(brier(prob_ov25, real_ov25), 4),
            "bss":      round(brier_skill(prob_ov25, real_ov25), 4),
            "tasa_real":   round(sum(real_ov25) / n, 3),
            "prob_modelo": round(sum(prob_ov25) / n, 3),
        },
        "btts": {
            "log_loss": round(log_loss(prob_btts, real_btts), 4),
            "brier":    round(brier(prob_btts, real_btts), 4),
            "bss":      round(brier_skill(prob_btts, real_btts), 4),
        },
        "1x2": {
            "brier_local":    round(brier(prob_1, real_1), 4),
            "brier_empate":   round(brier(prob_x, real_x), 4),
            "brier_visitante":round(brier(prob_2, real_2), 4),
            "bss_local":      round(brier_skill(prob_1, real_1), 4),
            "bss_empate":     round(brier_skill(prob_x, real_x), 4),
            "bss_visitante":  round(brier_skill(prob_2, real_2), 4),
        },
    }

    if verbose:
        header(f"EVALUACIÓN xGoals — N={n} partidos")
        xg = resultados["xgoals"]
        section("xG vs goles reales")
        row("Local     MAE",  f"{xg['mae_local']:.3f}  RMSE={xg['rmse_local']:.3f}  bias={xg['bias_local']:+.3f}")
        row("Visitante MAE",  f"{xg['mae_visitante']:.3f}  RMSE={xg['rmse_visitante']:.3f}  bias={xg['bias_visitante']:+.3f}")
        section("Probabilidades")
        ov = resultados["over25"]
        row("Over 2.5  Brier", f"{ov['brier']:.4f}  BSS={ov['bss']:+.4f}  tasa_real={ov['tasa_real']:.2f}  p_modelo={ov['prob_modelo']:.2f}")
        bt = resultados["btts"]
        row("BTTS      Brier", f"{bt['brier']:.4f}  BSS={bt['bss']:+.4f}")
        res = resultados["1x2"]
        row("1X2 Brier (L/X/2)", f"{res['brier_local']:.4f} / {res['brier_empate']:.4f} / {res['brier_visitante']:.4f}")
        row("1X2 BSS   (L/X/2)", f"{res['bss_local']:+.4f} / {res['bss_empate']:+.4f} / {res['bss_visitante']:+.4f}")
        print(f"\n  Referencia BSS: >0=mejor que base | >0.05=útil | >0.15=bueno")
        print(f"  Referencia Brier: 0.25=sin info (p=0.5 siempre)")
        footer()

    return resultados


def main():
    from transformation import calcular_xstyle
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli("Backtesting de xGoals")
    partidos = load_partidos()
    evaluar(partidos, calcular_xstyle(partidos), n_ultimos=args.n)


if __name__ == "__main__":
    main()
