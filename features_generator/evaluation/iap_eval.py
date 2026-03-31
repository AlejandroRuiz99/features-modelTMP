"""
iap_eval.py — Validación del IAP (Índice de Agresividad Ponderado) como predictor de faltas.

Responde a:
  1. ¿Correlaciona el IAP del partido (media de los dos equipos) con las faltas reales?
  2. ¿Cuáles son los pesos óptimos de (faltas, amarillas) en el índice?
  3. ¿Cuánto aporta el componente de amarillas vs. usar solo faltas históricas?

Calibra: PESO_FALTAS, PESO_AMARILLAS (PESO_ROJAS se fuerza a 0 por ser ruido).

Uso:
  python -m evaluate.iap_eval
  python -m evaluate.iap_eval --calibrar --update-config
"""

from __future__ import annotations

from core.helpers import parse_date, safe, decay_weight
from datetime import date as _date
from evaluation.base import pearson, mae, bias, header, section, row, footer

MIN_CONTEXT = 30
MIN_FOULS   = 5

# Grid: solo faltas y amarillas (rojas no discriminan bien)
PESO_FALTAS_GRID    = [0.5, 0.8, 1.0, 1.2]
PESO_AMARILLAS_GRID = [0.0, 0.1, 0.2, 0.35, 0.5]


def _iap_partido(partidos_contexto: list[dict], equipo: str, w_f: float, w_a: float) -> float:
    """IAP ponderado por decay para un equipo dado los partidos de contexto."""
    from core.config import DECAY_LAMBDA
    hoy = _date.today()
    num = den = 0.0
    for p in partidos_contexto:
        for rol in ("home", "away"):
            t = p[rol]
            if t["name"] != equipo:
                continue
            peso = decay_weight(parse_date(p["date"]), hoy, DECAY_LAMBDA)
            iap_raw = safe(t.get("fouls")) * w_f + safe(t.get("yellow_cards")) * w_a
            num += iap_raw * peso
            den += peso
    return num / den if den > 0 else 0.0


def _walk_forward(
    partidos_sorted: list[dict],
    muestra: list[dict],
    w_f: float,
    w_a: float,
) -> tuple[list[float], list[float]]:
    """Devuelve (iap_medios_predichos, faltas_totales_reales)."""
    iap_preds, reals = [], []
    for partido in muestra:
        ea = partido["home"]["name"]
        eb = partido["away"]["name"]
        f_l = safe(partido["home"].get("fouls"))
        f_v = safe(partido["away"].get("fouls"))
        if f_l + f_v < MIN_FOULS:
            continue
        contexto = [p for p in partidos_sorted if p["date"] < partido["date"]]
        if len(contexto) < MIN_CONTEXT:
            continue

        iap_a = _iap_partido(contexto, ea, w_f, w_a)
        iap_b = _iap_partido(contexto, eb, w_f, w_a)
        iap_preds.append((iap_a + iap_b) / 2)
        reals.append(float(f_l + f_v))

    return iap_preds, reals


def evaluar(
    partidos: list[dict],
    n_ultimos: int = 150,
    verbose: bool = True,
) -> dict:
    """Backtesting del IAP con los pesos actuales de config.yaml."""
    from core.config import PESO_FALTAS, PESO_AMARILLAS

    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:]
    preds, reals = _walk_forward(partidos_sorted, muestra, PESO_FALTAS, PESO_AMARILLAS)

    if not preds:
        if verbose:
            print("[WARN] Sin datos suficientes para evaluar IAP.")
        return {}

    r = pearson(preds, reals)
    resultados = {
        "n": len(preds),
        "params": {"peso_faltas": PESO_FALTAS, "peso_amarillas": PESO_AMARILLAS},
        "correlacion_r": round(r, 3),
        "mae": round(mae(preds, reals), 3),
        "bias": round(bias(preds, reals), 3),
    }

    if verbose:
        header(f"EVALUACIÓN IAP — N={len(preds)} partidos")
        row("Pesos actuales", f"faltas={PESO_FALTAS}  amarillas={PESO_AMARILLAS}")
        row("Correlación Pearson r", f"{r:.3f}  [ref: >0.25=señal | >0.50=buena]")
        row("MAE (IAP vs fouls)", f"{resultados['mae']:.3f}  (escala IAP, no fouls directos)")
        row("Bias", f"{resultados['bias']:+.3f}")
        footer()

    return resultados


def calibrar(
    partidos: list[dict],
    n_ultimos: int = 150,
    verbose: bool = True,
) -> dict:
    """Grid search sobre pesos IAP maximizando correlación con faltas reales."""
    partidos_sorted = sorted(partidos, key=lambda p: p["date"])
    muestra = partidos_sorted[-n_ultimos:]

    best_r = -float("inf")
    best   = {"peso_faltas": 1.0, "peso_amarillas": 0.35}
    grid   = []

    for w_f in PESO_FALTAS_GRID:
        for w_a in PESO_AMARILLAS_GRID:
            preds, reals = _walk_forward(partidos_sorted, muestra, w_f, w_a)
            if len(preds) < 10:
                continue
            r = pearson(preds, reals)
            grid.append({"peso_faltas": w_f, "peso_amarillas": w_a, "r": r, "n": len(preds)})
            if r > best_r:
                best_r = r
                best = {"peso_faltas": w_f, "peso_amarillas": w_a}

    if verbose:
        header("CALIBRACIÓN IAP — Grid Search de pesos")
        print(f"  {'w_faltas':>9}  {'w_amarillas':>11}  {'r':>8}  {'n':>5}")
        for g in sorted(grid, key=lambda x: -x["r"]):
            marker = " ◀ MEJOR" if (g["peso_faltas"] == best["peso_faltas"] and g["peso_amarillas"] == best["peso_amarillas"]) else ""
            print(f"  {g['peso_faltas']:>9.2f}  {g['peso_amarillas']:>11.2f}  {g['r']:>8.4f}  {g['n']:>5}{marker}")
        footer()

    return {"best_peso_faltas": best["peso_faltas"], "best_peso_amarillas": best["peso_amarillas"], "best_r": best_r, "grid": grid}


def main():
    from evaluation.base import load_partidos, eval_cli

    args = eval_cli("Validación de los pesos IAP", calibrable=True)
    partidos = load_partidos()

    evaluar(partidos, n_ultimos=args.n)

    if args.calibrar or args.update_config:
        cal = calibrar(partidos, n_ultimos=args.n)
        if args.update_config:
            _update_config(cal["best_peso_faltas"], cal["best_peso_amarillas"])
            print(f"\n[OK] config.yaml actualizado con peso_faltas={cal['best_peso_faltas']}, peso_amarillas={cal['best_peso_amarillas']}")


def _update_config(peso_faltas: float, peso_amarillas: float) -> None:
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("pesos", {})
    cfg["pesos"]["faltas"]    = peso_faltas
    cfg["pesos"]["amarillas"] = peso_amarillas
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
