"""
referee_eval.py — Validación del factor árbitro en la predicción de faltas.

Responde a:
  1. ¿Añade valor el perfil del árbitro? (modelo con vs. sin factor árbitro)
  2. ¿Qué árbitros tienen el perfil más estable/inestable?
  3. ¿A partir de cuántos partidos el perfil del árbitro es fiable?

Uso:
  python -m evaluate.referee_eval
  python -m evaluate.referee_eval --n 150
"""

from __future__ import annotations

from core.helpers import safe
from transformation.xfouls import calcular_xfouls
from transformation.referees import calcular_perfiles
from evaluation.base import mae, bias, pearson, header, section, row, footer

MIN_CONTEXT = 30
MIN_FOULS   = 5


def evaluar(
    partidos: list[dict],
    n_ultimos: int = 150,
    verbose: bool = True,
) -> dict:
    """
    Compara MAE de xFouls con y sin factor árbitro en walk-forward.
    También analiza la fiabilidad del perfil por nº de partidos.
    """
    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:]

    preds_con_ref,   reals_con    = [], []
    preds_sin_ref,   reals_sin    = [], []
    ref_n_partidos                = []   # nº partidos del árbitro en el contexto
    by_ref: dict[str, dict]       = {}   # métricas por árbitro

    for partido in muestra:
        ea     = partido["home"]["name"]
        eb     = partido["away"]["name"]
        f_l    = safe(partido["home"].get("fouls"))
        f_v    = safe(partido["away"].get("fouls"))
        f_real = f_l + f_v
        if f_real < MIN_FOULS:
            continue

        arbitro = partido.get("referee")
        contexto = [p for p in partidos_sorted if p["date"] < partido["date"]]
        if len(contexto) < MIN_CONTEXT:
            continue

        xf_con  = calcular_xfouls(contexto, ea, eb, arbitro=arbitro)
        xf_sin  = calcular_xfouls(contexto, ea, eb, arbitro=None)

        preds_con_ref.append(xf_con["xfouls_total"])
        reals_con.append(float(f_real))
        preds_sin_ref.append(xf_sin["xfouls_total"])
        reals_sin.append(float(f_real))

        # Nº de partidos del árbitro en el contexto
        if arbitro:
            perfiles = calcular_perfiles(contexto)
            n_ref = perfiles.get(arbitro, {}).get("partidos", 0)
            ref_n_partidos.append(n_ref)

            if arbitro not in by_ref:
                by_ref[arbitro] = {"preds": [], "reals": [], "n_perfil": n_ref}
            by_ref[arbitro]["preds"].append(xf_con["xfouls_total"])
            by_ref[arbitro]["reals"].append(float(f_real))

    n = len(preds_con_ref)
    if n == 0:
        if verbose:
            print("[WARN] Sin datos suficientes para evaluar referee_eval.")
        return {}

    mae_con = mae(preds_con_ref, reals_con)
    mae_sin = mae(preds_sin_ref, reals_sin)
    mejora  = mae_sin - mae_con   # positivo = el árbitro ayuda

    # Fiabilidad: dividir por nº de partidos del árbitro en el contexto
    buckets = {
        "sin_perfil": ([], []),
        "1-10":       ([], []),
        "11-30":      ([], []),
        ">30":        ([], []),
    }
    for i, n_ref in enumerate(ref_n_partidos):
        f_real = reals_con[i]
        p_con  = preds_con_ref[i]
        if n_ref == 0:
            buckets["sin_perfil"][0].append(p_con); buckets["sin_perfil"][1].append(f_real)
        elif n_ref <= 10:
            buckets["1-10"][0].append(p_con);       buckets["1-10"][1].append(f_real)
        elif n_ref <= 30:
            buckets["11-30"][0].append(p_con);      buckets["11-30"][1].append(f_real)
        else:
            buckets[">30"][0].append(p_con);        buckets[">30"][1].append(f_real)

    resultados = {
        "n": n,
        "mae_con_arbitro": round(mae_con, 3),
        "mae_sin_arbitro": round(mae_sin, 3),
        "mejora_mae":      round(mejora, 3),
        "bias_con":        round(bias(preds_con_ref, reals_con), 3),
        "bias_sin":        round(bias(preds_sin_ref, reals_sin), 3),
        "r_con":           round(pearson(preds_con_ref, reals_con), 3),
        "fiabilidad_por_n": {
            k: {"mae": round(mae(v[0], v[1]), 3), "n": len(v[0])}
            for k, v in buckets.items() if v[0]
        },
    }

    if verbose:
        header(f"EVALUACIÓN Factor Árbitro — N={n} partidos")
        section("Impacto del factor árbitro")
        row("MAE con árbitro",   f"{mae_con:.3f}")
        row("MAE sin árbitro",   f"{mae_sin:.3f}")
        row("Mejora (↓ MAE)",    f"{mejora:+.3f}  {'✓ árbitro añade valor' if mejora > 0 else '✗ árbitro no añade valor'}")
        row("Correlación r (con)", f"{resultados['r_con']:.3f}")

        section("Fiabilidad por nº partidos del árbitro")
        for k, v in resultados["fiabilidad_por_n"].items():
            row(f"  n partidos árbitro {k}", f"MAE={v['mae']:.3f}  ({v['n']} obs.)")

        print(f"\n  Interpretación: si MAE mejora con más partidos del árbitro,")
        print(f"  el perfil converge correctamente con el histórico.")
        footer()

    return resultados


def main():
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli("Validación del factor árbitro")
    evaluar(load_partidos(), n_ultimos=args.n)


if __name__ == "__main__":
    main()
