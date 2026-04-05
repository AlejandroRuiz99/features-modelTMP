#!/usr/bin/env python3
"""
backtest.py — Evaluación retrospectiva del ensemble de predicción de faltas.

Métricas por modelo (ensemble, NB, NegBin, ANFIS):
  - MAE   : error absoluto medio sobre la media predicha
  - NLL   : negative log-likelihood (log-loss discreto)
  - CRPS  : Continuous Ranked Probability Score (discreto, suma sobre k=0..60)
  - Brier : error cuadrático medio en P(over línea) para líneas clave

Salida: reporte HTML con tablas y gráficas de calibración.

Uso:
  python scripts/backtest.py
  python scripts/backtest.py --checkpoint prediction_models/checkpoints/ensemble
  python scripts/backtest.py --data prediction_models/data/training.parquet --output backtest_report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# sys.path: habilita imports sin prefijo de paquete (igual que run_prediction.py)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "prediction_models"))
sys.path.insert(0, str(_ROOT / "features_generator"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = _ROOT / "prediction_models" / "checkpoints" / "ensemble"
DEFAULT_DATA = _ROOT / "prediction_models" / "data" / "training.parquet"
DEFAULT_OUTPUT = _ROOT / "backtest_report.html"
BRIER_LINES = [21.5, 23.5, 25.5, 27.5, 29.5, 31.5, 33.5]
MODEL_KEYS = ["ensemble", "nb", "regression", "anfis"]
MODEL_LABELS = {
    "ensemble": "Ensemble (final)",
    "nb": "Naive Bayes",
    "regression": "NegBin Regressor",
    "anfis": "ANFIS",
}
PMF_MAX_K = 60  # FoulPMF cubre 0..60


# ---------------------------------------------------------------------------
# Métricas por distribución
# ---------------------------------------------------------------------------

def _nll(pmf_probs: np.ndarray, observed: int) -> float:
    k = max(0, min(observed, PMF_MAX_K))
    p = float(pmf_probs[k])
    return -np.log(max(p, 1e-12))


def _crps(cdf: np.ndarray, observed: int) -> float:
    """CRPS discreto: sum_{k=0}^{60} (CDF(k) - 1{k >= observed})^2."""
    indicator = np.zeros(len(cdf))
    obs = max(0, min(observed, PMF_MAX_K))
    indicator[obs:] = 1.0
    return float(np.sum((cdf - indicator) ** 2))


def _brier_at_line(pmf_prob_over: float, observed: int, line: float) -> float:
    outcome = 1.0 if observed > line else 0.0
    return (pmf_prob_over - outcome) ** 2


# ---------------------------------------------------------------------------
# Estructura de resultados por partido
# ---------------------------------------------------------------------------

@dataclass
class RowResult:
    season: str
    observed: int
    # predicted means
    mean: dict[str, float] = field(default_factory=dict)
    # scalar metrics
    nll: dict[str, float] = field(default_factory=dict)
    crps: dict[str, float] = field(default_factory=dict)
    mae: dict[str, float] = field(default_factory=dict)
    # brier per line: {model: {line: score}}
    brier: dict[str, dict[float, float]] = field(default_factory=dict)
    # calibration raw: {model: {line: (p_over, outcome)}}
    calib: dict[str, dict[float, tuple[float, float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core: predecir y calcular métricas para una fila
# ---------------------------------------------------------------------------

def _row_metrics(row: dict, ensemble: Any) -> RowResult | None:
    observed_total = row.get("fouls_total")
    if observed_total is None:
        return None
    observed = int(observed_total)
    season = str(row.get("season", "unknown"))

    try:
        result = ensemble.predict(row)
    except Exception as e:
        logger.debug("predict() failed: %s", e)
        return None

    pmfs = {
        "ensemble": result.pmf_total,
        "nb": result.pmf_bayes,
        "regression": result.pmf_regression,
        "anfis": result.pmf_anfis,
    }

    rr = RowResult(season=season, observed=observed)
    for key, pmf in pmfs.items():
        probs = np.array(pmf.probs) if hasattr(pmf, "probs") else np.array(pmf._probs)
        cdf = np.array(pmf.cdf)
        mu = float(pmf.mean)

        rr.mean[key] = mu
        rr.nll[key] = _nll(probs, observed)
        rr.crps[key] = _crps(cdf, observed)
        rr.mae[key] = abs(mu - observed)

        rr.brier[key] = {}
        rr.calib[key] = {}
        for line in BRIER_LINES:
            p_over = float(pmf.prob_over(line))
            rr.brier[key][line] = _brier_at_line(p_over, observed, line)
            rr.calib[key][line] = (p_over, 1.0 if observed > line else 0.0)

    return rr


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------

def _aggregate(results: list[RowResult]) -> dict:
    seasons = sorted(set(r.season for r in results))

    def _stats(subset: list[RowResult], key: str) -> dict[str, float]:
        if not subset:
            return {}
        return {
            "mae": float(np.mean([r.mae[key] for r in subset])),
            "nll": float(np.mean([r.nll[key] for r in subset])),
            "crps": float(np.mean([r.crps[key] for r in subset])),
            "brier_mean": float(
                np.mean([np.mean(list(r.brier[key].values())) for r in subset])
            ),
            "n": len(subset),
        }

    return {
        "global": {k: _stats(results, k) for k in MODEL_KEYS},
        "by_season": {
            s: {k: _stats([r for r in results if r.season == s], k) for k in MODEL_KEYS}
            for s in seasons
        },
        "seasons": seasons,
        "n_total": len(results),
    }


def _calibration_data(results: list[RowResult]) -> dict[str, dict[float, dict]]:
    """Para cada modelo y línea: buckets de calibración {bin_center: {mean_pred, freq_actual, n}}."""
    n_bins = 10
    out: dict[str, dict[float, dict]] = {}
    for key in MODEL_KEYS:
        out[key] = {}
        for line in BRIER_LINES:
            bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
            for r in results:
                p, outcome = r.calib[key][line]
                b = min(int(p * n_bins), n_bins - 1)
                bins[b].append((p, outcome))
            out[key][line] = {
                "bins": [
                    {
                        "center": (i + 0.5) / n_bins,
                        "mean_pred": float(np.mean([x[0] for x in b])) if b else (i + 0.5) / n_bins,
                        "freq_actual": float(np.mean([x[1] for x in b])) if b else float("nan"),
                        "n": len(b),
                    }
                    for i, b in enumerate(bins)
                ]
            }
    return out


# ---------------------------------------------------------------------------
# Gráficas → base64
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_calibration(calib: dict, line: float) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfecta")
    colors = {"ensemble": "#1f77b4", "nb": "#ff7f0e", "regression": "#2ca02c", "anfis": "#d62728"}
    for key in MODEL_KEYS:
        pts = calib[key][line]["bins"]
        xs = [p["mean_pred"] for p in pts if not np.isnan(p["freq_actual"])]
        ys = [p["freq_actual"] for p in pts if not np.isnan(p["freq_actual"])]
        ax.plot(xs, ys, "o-", color=colors[key], label=MODEL_LABELS[key], ms=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("P(over) predicha")
    ax.set_ylabel("Frecuencia real")
    ax.set_title(f"Calibración — Over {line}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _plot_mae_by_season(agg: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seasons = agg["seasons"]
    x = np.arange(len(seasons))
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(max(6, len(seasons) * 1.2), 4))
    for i, key in enumerate(MODEL_KEYS):
        maes = [agg["by_season"][s][key].get("mae", float("nan")) for s in seasons]
        ax.bar(x + i * width, maes, width, label=MODEL_LABELS[key], color=colors[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(seasons, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE (faltas)")
    ax.set_title("MAE por temporada y modelo")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _plot_weights_dist(results: list[RowResult]) -> str | None:
    """Histograma de pesos del gating si están disponibles."""
    # Los pesos no están en RowResult — no se pueden graficar sin acceso al ensemble.
    return None


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Backtest — foultsPredictor</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; margin-top: 40px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th {{ background: #16213e; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f9f9f9; }}
  .best {{ font-weight: bold; color: #1a7a4a; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; margin-top: 20px; }}
  img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
  .meta {{ background: #f5f5f5; padding: 14px 18px; border-radius: 8px; font-size: 13px; margin-bottom: 30px; }}
  .metric-label {{ font-size: 11px; color: #666; display: block; }}
</style>
</head>
<body>
<h1>Backtest — foultsPredictor</h1>
<div class="meta">
  <strong>Partidos evaluados:</strong> {n_total} &nbsp;|&nbsp;
  <strong>Temporadas:</strong> {seasons_str} &nbsp;|&nbsp;
  <strong>Checkpoint:</strong> {checkpoint} &nbsp;|&nbsp;
  <strong>Generado:</strong> {timestamp}
</div>

<h2>Métricas globales</h2>
{global_table}

<h2>MAE por temporada</h2>
<img src="data:image/png;base64,{mae_chart}" alt="MAE por temporada">

<h2>Métricas por temporada — Ensemble</h2>
{season_table}

<h2>Calibración</h2>
<div class="grid">
{calib_charts}
</div>

</body></html>
"""


def _global_table(agg: dict) -> str:
    rows_html = ""
    metrics = [("mae", "MAE ↓", "faltas"), ("nll", "NLL ↓", "nats"), ("crps", "CRPS ↓", ""), ("brier_mean", "Brier medio ↓", "")]
    best = {m: min(agg["global"][k].get(m, float("inf")) for k in MODEL_KEYS) for m, *_ in metrics}

    header = "<tr><th>Modelo</th>" + "".join(f"<th>{lbl}<br><span class='metric-label'>{unit}</span></th>" for _, lbl, unit in metrics) + "<th>N</th></tr>"
    for key in MODEL_KEYS:
        g = agg["global"][key]
        cells = ""
        for m, _, _ in metrics:
            v = g.get(m, float("nan"))
            cls = " class='best'" if abs(v - best[m]) < 1e-6 else ""
            cells += f"<td{cls}>{v:.4f}</td>"
        rows_html += f"<tr><td>{MODEL_LABELS[key]}</td>{cells}<td>{g.get('n', 0)}</td></tr>"
    return f"<table>{header}{rows_html}</table>"


def _season_table(agg: dict) -> str:
    header = "<tr><th>Temporada</th><th>N</th><th>MAE ↓</th><th>NLL ↓</th><th>CRPS ↓</th><th>Brier ↓</th></tr>"
    rows_html = ""
    for s in agg["seasons"]:
        g = agg["by_season"][s]["ensemble"]
        rows_html += (
            f"<tr><td>{s}</td><td>{g.get('n', 0)}</td>"
            f"<td>{g.get('mae', float('nan')):.4f}</td>"
            f"<td>{g.get('nll', float('nan')):.4f}</td>"
            f"<td>{g.get('crps', float('nan')):.4f}</td>"
            f"<td>{g.get('brier_mean', float('nan')):.4f}</td></tr>"
        )
    return f"<table>{header}{rows_html}</table>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest del ensemble de faltas")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Directorio del checkpoint")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Parquet de training")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Ruta del reporte HTML")
    p.add_argument("--max-rows", type=int, default=0, help="Limitar filas para prueba rápida (0 = todas)")
    p.add_argument("--calib-line", type=float, default=25.5, help="Línea OU para calibración detallada en el reporte")
    return p.parse_args()


def main() -> None:
    from datetime import datetime

    import pandas as pd
    from src.models.ensemble import FoulPredictionEnsemble

    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        logger.error("Checkpoint no encontrado: %s", checkpoint_path)
        sys.exit(1)
    if not data_path.exists():
        logger.error("Parquet no encontrado: %s", data_path)
        sys.exit(1)

    # -- Cargar ensemble --
    logger.info("Cargando ensemble desde %s ...", checkpoint_path)
    config_json = checkpoint_path / "config.json"
    config = json.load(open(config_json)) if config_json.exists() else {}
    ensemble = FoulPredictionEnsemble(config)
    ensemble.load(checkpoint_path)
    logger.info("Ensemble cargado.")

    # -- Cargar datos --
    logger.info("Cargando Parquet: %s", data_path)
    df = pd.read_parquet(data_path)
    if args.max_rows > 0:
        df = df.head(args.max_rows)
    logger.info("  %d filas x %d cols", len(df), len(df.columns))

    # -- Predicciones --
    results: list[RowResult] = []
    errors = 0
    for i, (_, row) in enumerate(df.iterrows(), 1):
        rr = _row_metrics(row.to_dict(), ensemble)
        if rr is not None:
            results.append(rr)
        else:
            errors += 1
        if i % 100 == 0 or i == len(df):
            logger.info("  %d/%d — OK:%d err:%d", i, len(df), len(results), errors)

    if not results:
        logger.error("Sin resultados válidos.")
        sys.exit(1)
    logger.info("Total válidos: %d  |  errores: %d", len(results), errors)

    # -- Agregar --
    agg = _aggregate(results)
    calib = _calibration_data(results)

    # -- Gráficas --
    logger.info("Generando gráficas...")
    mae_chart = _plot_mae_by_season(agg)
    calib_charts_html = ""
    for line in BRIER_LINES:
        b64 = _plot_calibration(calib, line)
        calib_charts_html += f'<div><img src="data:image/png;base64,{b64}" alt="Calibración {line}"></div>\n'

    # -- HTML --
    html = _HTML_TEMPLATE.format(
        n_total=agg["n_total"],
        seasons_str=", ".join(agg["seasons"]),
        checkpoint=str(checkpoint_path),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        global_table=_global_table(agg),
        mae_chart=mae_chart,
        season_table=_season_table(agg),
        calib_charts=calib_charts_html,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Reporte guardado en: %s", output_path)


if __name__ == "__main__":
    main()
