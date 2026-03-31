"""
xposesion_eval.py — Backtesting de xPosesión vs posesión real.

Solo evalúa partidos donde el dato real de posesión está disponible en Supabase.

Uso:
  python -m evaluate.xposesion_eval
"""

from __future__ import annotations

from transformation.xposesion import calcular_xposesion
from evaluation.base import mae, bias, pearson, header, section, row, footer

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

    for partido in muestra:
        ea = partido["home"]["name"]
        eb = partido["away"]["name"]
        if ea not in xstyles or eb not in xstyles:
            continue

        poss_l = partido["home"].get("possession")
        poss_v = partido["away"].get("possession")
        if poss_l is None or poss_v is None:
            continue

        contexto = [p for p in partidos_sorted if p["date"] < partido["date"]]
        if len(contexto) < MIN_CONTEXT:
            continue

        xp = calcular_xposesion(xstyles, ea, eb)
        preds_l.append(xp["posesion_local"])
        reals_l.append(float(poss_l))

    n = len(preds_l)
    if n == 0:
        if verbose:
            print("[WARN] Sin datos de posesión real disponibles para evaluar.")
        return {}

    resultados = {
        "n": n,
        "mae_pp":  round(mae(preds_l, reals_l), 2),
        "bias_pp": round(bias(preds_l, reals_l), 2),
        "r":       round(pearson(preds_l, reals_l), 3),
    }

    if verbose:
        header(f"EVALUACIÓN xPosesión — N={n} partidos con dato real")
        row("MAE",  f"{resultados['mae_pp']:.2f} pp  [ref: <5pp=útil | >10pp=sin valor]")
        row("Bias", f"{resultados['bias_pp']:+.2f} pp  [+ = sobreestima posesión local]")
        row("Correlación Pearson r", f"{resultados['r']:.3f}")
        print(f"\n  Nota: solo partidos con posesión real disponible en Supabase.")
        footer()

    return resultados


def main():
    from transformation import calcular_xstyle
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli("Backtesting de xPosesión")
    partidos = load_partidos()
    evaluar(partidos, calcular_xstyle(partidos), n_ultimos=args.n)


if __name__ == "__main__":
    main()
