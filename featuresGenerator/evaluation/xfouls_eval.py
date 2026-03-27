"""
xfouls_eval.py — Validación y calibración de la fórmula xFouls.

Responde a:
  1. ¿Cuán precisa es la fórmula xFouls comparada con las faltas reales?
  2. ¿Cuáles son los valores óptimos de DECAY_LAMBDA y ALPHA_CARD_PRESSURE?

Métodos:
  - Walk-forward backtesting: para cada partido, usa solo partidos anteriores como contexto.
  - Grid search 2D sobre (decay_lambda, alpha_card_pressure) minimizando MAE total.

Uso:
  python -m evaluate.xfouls_eval                     # con los 150 últimos partidos
  python -m evaluate.xfouls_eval --n 200             # con los 200 últimos partidos
  python -m evaluate.xfouls_eval --update-config     # actualiza config.yaml si mejora
"""

from __future__ import annotations

from core.helpers import safe
from transformation.xfouls import calcular_xfouls
from evaluation.base import mae, rmse, bias, pearson, header, section, row, footer

# Grid de búsqueda
DECAY_GRID = [0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020]
ALPHA_GRID = [0.0, 0.1, 0.2, 0.30, 0.50, 0.70, 1.0]

MIN_CONTEXT = 30   # partidos mínimos de contexto para una predicción válida
MIN_FOULS   = 5    # filtrar partidos con datos incompletos


# ---------------------------------------------------------------------------
# Walk-forward para una combinación fija de parámetros
# ---------------------------------------------------------------------------

def _walk_forward(
    partidos_sorted: list[dict],
    muestra: list[dict],
    decay_lambda: float,
    alpha_card_pressure: float,
) -> dict:
    """
    Ejecuta el backtesting walk-forward y devuelve las listas de
    predicciones y valores reales para calcular métricas.
    """
    preds_total, reals_total = [], []
    preds_local, reals_local = [], []
    preds_vis,   reals_vis   = [], []

    for partido in muestra:
        ea = partido["home"]["name"]
        eb = partido["away"]["name"]
        f_l = safe(partido["home"].get("fouls"))
        f_v = safe(partido["away"].get("fouls"))
        if f_l + f_v < MIN_FOULS:
            continue

        fecha_str = partido["date"]
        contexto = [p for p in partidos_sorted if p["date"] < fecha_str]
        if len(contexto) < MIN_CONTEXT:
            continue

        arbitro = partido.get("referee")
        xf = calcular_xfouls(
            contexto, ea, eb,
            arbitro=arbitro,
            alpha_card_pressure=alpha_card_pressure,
            _decay_lambda_override=decay_lambda,
        )

        preds_total.append(xf["xfouls_total"])
        reals_total.append(float(f_l + f_v))
        preds_local.append(xf["xfouls_local"])
        reals_local.append(float(f_l))
        preds_vis.append(xf["xfouls_visitante"])
        reals_vis.append(float(f_v))

    return {
        "preds_total": preds_total,
        "reals_total": reals_total,
        "preds_local": preds_local,
        "reals_local": reals_local,
        "preds_vis": preds_vis,
        "reals_vis": reals_vis,
        "n": len(preds_total),
    }


# ---------------------------------------------------------------------------
# Grid search 2D
# ---------------------------------------------------------------------------

def calibrar(
    partidos: list[dict],
    n_ultimos: int = 150,
    verbose: bool = True,
) -> dict:
    """
    Busca los valores óptimos de DECAY_LAMBDA y ALPHA_CARD_PRESSURE
    minimizando el MAE del total de faltas en un walk-forward.

    Devuelve dict con best_decay, best_alpha, best_mae, full_grid.
    """
    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:]

    if verbose:
        header("CALIBRACIÓN xFouls — Grid Search 2×2D")
        print(f"  Muestra: {len(muestra)} partidos | Mínimo contexto: {MIN_CONTEXT}")
        print(f"  Grid: {len(DECAY_GRID)} lambdas × {len(ALPHA_GRID)} alphas = {len(DECAY_GRID) * len(ALPHA_GRID)} combinaciones")

    grid: list[dict] = []
    best_mae_val = float("inf")
    best = {"decay_lambda": None, "alpha_card_pressure": None}

    for lam in DECAY_GRID:
        for alpha in ALPHA_GRID:
            res = _walk_forward(partidos_sorted, muestra, lam, alpha)
            if res["n"] < 10:
                continue
            mae_val = mae(res["preds_total"], res["reals_total"])
            grid.append({"decay_lambda": lam, "alpha_card_pressure": alpha, "mae_total": mae_val, "n": res["n"]})
            if mae_val < best_mae_val:
                best_mae_val = mae_val
                best = {"decay_lambda": lam, "alpha_card_pressure": alpha}

    if verbose:
        section("Resultados del grid (MAE total por combinación)")
        header_row = f"  {'lambda':>8}  {'alpha':>6}  {'MAE':>7}  {'n':>5}"
        print(header_row)
        for g in sorted(grid, key=lambda x: x["mae_total"]):
            marker = " ◀ MEJOR" if (g["decay_lambda"] == best["decay_lambda"] and g["alpha_card_pressure"] == best["alpha_card_pressure"]) else ""
            print(f"  {g['decay_lambda']:>8.3f}  {g['alpha_card_pressure']:>6.2f}  {g['mae_total']:>7.3f}  {g['n']:>5}{marker}")

    return {
        "best_decay_lambda": best["decay_lambda"],
        "best_alpha_card_pressure": best["alpha_card_pressure"],
        "best_mae": best_mae_val,
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Backtesting con parámetros actuales
# ---------------------------------------------------------------------------

def evaluar(
    partidos: list[dict],
    n_ultimos: int = 150,
    verbose: bool = True,
) -> dict:
    """
    Backtesting walk-forward con los parámetros actuales de config.yaml.
    Devuelve métricas de MAE, RMSE, bias y correlación.
    """
    from core.config import DECAY_LAMBDA, ALPHA_CARD_PRESSURE

    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:]
    res = _walk_forward(partidos_sorted, muestra, DECAY_LAMBDA, ALPHA_CARD_PRESSURE)

    if res["n"] == 0:
        if verbose:
            print("[WARN] Sin datos suficientes para evaluar xFouls.")
        return {}

    resultados = {
        "n": res["n"],
        "params": {"decay_lambda": DECAY_LAMBDA, "alpha_card_pressure": ALPHA_CARD_PRESSURE},
        "total": {
            "mae":  round(mae(res["preds_total"], res["reals_total"]), 3),
            "rmse": round(rmse(res["preds_total"], res["reals_total"]), 3),
            "bias": round(bias(res["preds_total"], res["reals_total"]), 3),
            "r":    round(pearson(res["preds_total"], res["reals_total"]), 3),
        },
        "local": {
            "mae":  round(mae(res["preds_local"], res["reals_local"]), 3),
            "bias": round(bias(res["preds_local"], res["reals_local"]), 3),
        },
        "visitante": {
            "mae":  round(mae(res["preds_vis"], res["reals_vis"]), 3),
            "bias": round(bias(res["preds_vis"], res["reals_vis"]), 3),
        },
    }

    if verbose:
        header(f"EVALUACIÓN xFouls — N={res['n']} partidos (walk-forward)")
        row("Params", f"λ={DECAY_LAMBDA}  α={ALPHA_CARD_PRESSURE}")
        section("Total faltas")
        row("MAE",  f"{resultados['total']['mae']:.3f}  [ref: <4.0=útil | >6.0=sin valor]")
        row("RMSE", f"{resultados['total']['rmse']:.3f}")
        row("Bias", f"{resultados['total']['bias']:+.3f}  [+ = sobreestima]")
        row("Correlación Pearson r", f"{resultados['total']['r']:.3f}  [ref: >0.30=señal]")
        section("Por equipo")
        row("Local   MAE", f"{resultados['local']['mae']:.3f}  bias={resultados['local']['bias']:+.3f}")
        row("Visitante MAE", f"{resultados['visitante']['mae']:.3f}  bias={resultados['visitante']['bias']:+.3f}")
        footer()

    return resultados


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli("Validación y calibración de xFouls", calibrable=True)
    partidos = load_partidos()

    resultados_actuales = evaluar(partidos, n_ultimos=args.n)

    if args.calibrar or args.update_config:
        cal = calibrar(partidos, n_ultimos=args.n)

        if args.update_config and cal["best_mae"] < resultados_actuales.get("total", {}).get("mae", float("inf")):
            _update_config(cal["best_decay_lambda"], cal["best_alpha_card_pressure"])
            print(f"\n[OK] config.yaml actualizado con λ={cal['best_decay_lambda']}, α={cal['best_alpha_card_pressure']}")
        elif args.update_config:
            print("\n[INFO] Los parámetros actuales ya son óptimos. config.yaml sin cambios.")


def _update_config(decay_lambda: float, alpha_card_pressure: float) -> None:
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["decay_lambda"] = decay_lambda
    cfg["alpha_card_pressure"] = alpha_card_pressure
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
