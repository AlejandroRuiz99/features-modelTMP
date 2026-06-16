#!/usr/bin/env python3
"""
analyze_backtest.py — Análisis estadístico completo de backtest de apuestas.

Carga uno o más CSVs generados por backtest_odds.py y produce:
  - Métricas globales (hit rate, ROI, Sharpe, max drawdown, profit factor)
  - Test de Poisson-Binomial (H0: el modelo no bate al mercado)
  - Bootstrap 95% CI en ROI (10 000 remuestreos)
  - Reliability diagram + Brier Score (calibración del modelo)
  - Segmentación: fuente, mercado, lado, edge bucket, equipo, jornada
  - Curva PnL acumulada por jornada
  - Reporte Markdown en reports/analysis_{label}.md

Uso:
    python scripts/analyze_backtest.py reports/backtest_codere_e05.csv
    python scripts/analyze_backtest.py reports/backtest_all_e05.csv --label all_e05
    python scripts/analyze_backtest.py reports/backtest_codere_e05.csv reports/backtest_all_e05.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _detect_delimiter(path: Path) -> str:
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


def _f(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _int(v: Any) -> int | None:
    f = _f(v)
    return int(round(f)) if f is not None else None


def load_csv(path: Path) -> list[dict[str, Any]]:
    delim = _detect_delimiter(path)
    bets: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            p_over = _f(r.get("p_over_model"))
            edge = _f(r.get("edge"))
            odds = _f(r.get("odds"))
            won = _int(r.get("won"))
            pnl = _f(r.get("pnl"))
            bet_side = (r.get("bet") or "").strip()

            if p_over is None or edge is None or odds is None or won is None:
                continue

            # PnL: recompute if missing to avoid stale data
            if pnl is None:
                pnl = (odds - 1.0) if won else -1.0

            p_model_correct = p_over if bet_side == "over" else (1.0 - p_over)

            bets.append(
                {
                    "fixture_id": r.get("fixture_id", ""),
                    "date": r.get("date", ""),
                    "jornada": _int(r.get("jornada")),
                    "home": (r.get("home") or "").strip(),
                    "away": (r.get("away") or "").strip(),
                    "fuente": (r.get("fuente") or "").strip(),
                    "mercado": (r.get("mercado") or "").strip(),
                    "side_label": (r.get("side_label") or "").strip(),
                    "linea": _f(r.get("linea")),
                    "p_over_model": p_over,
                    "p_model_correct": p_model_correct,
                    "edge": edge,
                    "bet": bet_side,
                    "odds": odds,
                    "actual": _int(r.get("actual")),
                    "won": won,
                    "pnl": pnl,
                }
            )
    return bets


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def compute_metrics(bets: list[dict]) -> dict:
    if not bets:
        return {"n": 0}

    n = len(bets)
    wins = sum(b["won"] for b in bets)
    pnls = [b["pnl"] for b in bets]
    total_pnl = sum(pnls)
    avg_odds = sum(b["odds"] for b in bets) / n
    hit_rate = wins / n
    roi = total_pnl / n

    std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 0.0
    sharpe = roi / std_pnl if std_pnl > 0 else 0.0

    # Max drawdown sobre la curva de PnL acumulado
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(np.maximum(0.0, cumulative))
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns))

    calmar = abs(roi / max_dd) if max_dd > 1e-9 else float("inf")

    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 1e-9 else float("inf")

    # Poisson-Binomial test: H0 cada apuesta se gana con prob = 1/odds
    p_nulls = [1.0 / b["odds"] for b in bets]
    e_wins = sum(p_nulls)
    var_wins = sum(p * (1.0 - p) for p in p_nulls)
    z_stat = (wins - e_wins) / math.sqrt(var_wins) if var_wins > 1e-9 else 0.0
    p_value = 1.0 - _norm_cdf(z_stat)  # cola superior: H1 = modelo bate al mercado

    return {
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "hit_rate": hit_rate,
        "avg_odds": avg_odds,
        "market_implied_rate": e_wins / n,
        "edge_vs_market": hit_rate - e_wins / n,
        "total_pnl": total_pnl,
        "roi": roi,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "z_stat": z_stat,
        "p_value": p_value,
        "e_wins": e_wins,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_roi_ci(
    bets: list[dict], n_resamples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pnls = np.array([b["pnl"] for b in bets])
    n = len(pnls)
    rois = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(pnls, size=n, replace=True)
        rois[i] = float(np.mean(sample))
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------

_CAL_BREAKS = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
_CAL_LABELS = ["[50-55%)", "[55-60%)", "[60-65%)", "[65-70%)", "[70%+)"]


def compute_calibration(bets: list[dict]) -> list[dict]:
    buckets: dict[str, list[int]] = {lbl: [] for lbl in _CAL_LABELS}
    for b in bets:
        p = b["p_model_correct"]
        for i in range(len(_CAL_LABELS)):
            if _CAL_BREAKS[i] <= p < _CAL_BREAKS[i + 1]:
                buckets[_CAL_LABELS[i]].append(b["won"])
                break

    result = []
    for i, lbl in enumerate(_CAL_LABELS):
        wins_list = buckets[lbl]
        nn = len(wins_list)
        if nn == 0:
            continue
        lo, hi = _CAL_BREAKS[i], min(_CAL_BREAKS[i + 1], 0.975)
        midpoint = (lo + hi) / 2
        obs = sum(wins_list) / nn
        diff = obs - midpoint
        calib = "✅" if abs(diff) < 0.06 else ("⚠️ " if abs(diff) < 0.12 else "❌")
        result.append(
            {
                "bucket": lbl,
                "n": nn,
                "expected_p": midpoint,
                "observed_p": obs,
                "diff": diff,
                "calib": calib,
            }
        )
    return result


def brier_score(bets: list[dict]) -> float:
    return float(np.mean([(b["p_model_correct"] - b["won"]) ** 2 for b in bets]))


# ---------------------------------------------------------------------------
# Segmentos
# ---------------------------------------------------------------------------


def segment_analysis(bets: list[dict], key: str) -> list[tuple[str, dict]]:
    groups: dict[str, list] = defaultdict(list)
    for b in bets:
        groups[str(b.get(key, "?"))].append(b)
    return [
        (k, compute_metrics(g))
        for k, g in sorted(groups.items(), key=lambda x: -len(x[1]))
    ]


def team_analysis(bets: list[dict], min_n: int = 5) -> list[tuple[str, dict]]:
    groups: dict[str, list] = defaultdict(list)
    for b in bets:
        for team in [b["home"], b["away"]]:
            if team:
                groups[team].append(b)
    return [(k, compute_metrics(g)) for k, g in groups.items() if len(g) >= min_n]


def edge_bucket_analysis(bets: list[dict]) -> list[tuple[str, dict]]:
    buckets = [(0.05, 0.07), (0.07, 0.10), (0.10, 0.13), (0.13, 0.16), (0.16, 1.0)]
    labels = ["[5-7%)", "[7-10%)", "[10-13%)", "[13-16%)", "[16%+)"]
    groups: dict[str, list] = defaultdict(list)
    for b in bets:
        e = b["edge"]
        for i, (lo, hi) in enumerate(buckets):
            if lo <= e < hi:
                groups[labels[i]].append(b)
                break
    return [(lbl, compute_metrics(groups[lbl])) for lbl in labels if lbl in groups]


def jornada_analysis(bets: list[dict]) -> list[tuple[int, dict]]:
    groups: dict[int, list] = defaultdict(list)
    for b in bets:
        j = b.get("jornada")
        if j is not None:
            groups[j].append(b)
    return sorted(
        [(j, compute_metrics(g)) for j, g in groups.items()], key=lambda x: x[0]
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def _ff(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def _sig(p: float) -> str:
    if p < 0.01:
        return "★★★ p<1%"
    if p < 0.05:
        return "★★  p<5%"
    if p < 0.10:
        return "★   p<10%"
    return "—   no sig"


def _min_n_for_sig(m: dict) -> float:
    roi = m.get("roi", 0)
    std = m.get("std_pnl", 1)
    if roi <= 0 or std <= 0:
        return float("inf")
    return (1.645 * std / roi) ** 2


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def generate_report(
    bets: list[dict],
    label: str,
    ci_lo: float,
    ci_hi: float,
    source_path: str = "",
) -> str:
    m = compute_metrics(bets)
    if m["n"] == 0:
        return f"# {label}\n\nSin apuestas cargadas.\n"

    cal = compute_calibration(bets)
    bs = brier_score(bets)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    L: list[str] = []

    def line(s: str = "") -> None:
        L.append(s + "\n")

    def h(n: int, t: str) -> None:
        line(f"{'#' * n} {t}")
        line()

    # ── Header ──────────────────────────────────────────────────────────────
    h(1, f"Certificación — foultsPredictor · `{label}`")
    line(f"_Generado: {ts}_  |  _Fuente: {source_path}_")
    line()

    # ── 1. Resumen ejecutivo ─────────────────────────────────────────────────
    h(2, "1. Resumen Ejecutivo")

    roi_ok = m["roi"] > 0.05
    sig_ok = m["p_value"] < 0.10
    ci_ok = ci_lo > -0.03
    n_ok = m["n"] >= 80

    if roi_ok and sig_ok and ci_ok:
        verdict = "✅ GO — evidencia estadística de edge positivo"
    elif roi_ok and (sig_ok or ci_ok):
        verdict = "🟡 PROMETEDOR — ROI positivo, significancia parcial, ampliar muestra"
    elif roi_ok:
        verdict = "🟡 BAJO VIGILANCIA — ROI positivo pero sin significancia aún"
    else:
        verdict = "❌ NO-GO — ROI negativo"

    line(f"**{verdict}**")
    line()
    line(
        f"| Indicador | Valor | Umbral | Estado |"
        "\n|-----------|-------|--------|--------|"
    )
    line(f"| ROI       | {_pct(m['roi'], 2)} | >5%  | {'✅' if roi_ok else '❌'} |")
    line(f"| p-valor   | {m['p_value']:.4f}  | <0.10 | {'✅' if sig_ok else '❌'} |")
    line(f"| IC 95% inferior | {_pct(ci_lo, 2)} | >-3% | {'✅' if ci_ok else '❌'} |")
    line(f"| Muestra ≥ 80 | {m['n']} | ≥80 | {'✅' if n_ok else '⚠️'} |")
    line()

    if not n_ok:
        needed = _min_n_for_sig(m)
        line(
            f"> ⚠️ **Muestra pequeña** ({m['n']} apuestas). "
            f"Para p<5% con el ROI observado se necesitan ≈ **{needed:.0f} apuestas**."
        )
        line()

    # ── 2. Métricas globales ─────────────────────────────────────────────────
    h(2, "2. Métricas Globales")
    line("| Métrica | Valor |")
    line("|---------|-------|")
    rows = [
        ("Apuestas (n)", str(m["n"])),
        ("Ganadas / Perdidas", f"{m['wins']} / {m['losses']}"),
        ("Hit rate observada", _pct(m["hit_rate"])),
        ("Tasa implícita mercado (vig-free)", _pct(m["market_implied_rate"])),
        ("Edge vs mercado (hit rate delta)", _pct(m["edge_vs_market"], 2)),
        ("Odds promedio", _ff(m["avg_odds"])),
        ("PnL total", f"{m['total_pnl']:+.2f} u"),
        ("ROI por apuesta", _pct(m["roi"], 2)),
        ("Std PnL (per bet)", _ff(m["std_pnl"])),
        ("Sharpe (per-bet, sin anualizacion)", _ff(m["sharpe"])),
        ("Max Drawdown", f"{m['max_drawdown']:.2f} u"),
        ("Calmar ratio (ROI/MaxDD)", _ff(m["calmar"])),
        ("Profit Factor", _ff(m["profit_factor"])),
        ("Brier Score", _ff(bs, 4)),
    ]
    for name, val in rows:
        line(f"| {name} | {val} |")
    line()

    # ── 3. Significancia estadística ─────────────────────────────────────────
    h(2, "3. Significancia Estadística")
    h(3, "3.1 Test de Poisson-Binomial")
    line(
        "H₀: para cada apuesta, la probabilidad de victoria = 1/odds (el modelo no bate al mercado).  "
    )
    line("H₁: hit_rate real > tasa vig-free (el modelo tiene edge).")
    line()
    line(f"- Victorias esperadas bajo H₀: **{m['e_wins']:.1f}**")
    line(f"- Victorias observadas: **{m['wins']}**")
    line(f"- Z-estadístico: **{m['z_stat']:.3f}**")
    line(f"- p-valor (cola superior): **{m['p_value']:.4f}**")
    line(f"- Significancia: **{_sig(m['p_value'])}**")
    line()

    h(3, "3.2 Bootstrap 95% CI sobre ROI (10 000 remuestreos)")
    line(f"- IC: **[{_pct(ci_lo, 2)}, {_pct(ci_hi, 2)}]**")
    if ci_lo > 0:
        line(
            "- ✅ El límite inferior es positivo → edge con alta probabilidad sostenido."
        )
    elif ci_lo > -0.03:
        line(
            "- ⚠️  El límite inferior roza cero → se necesita más muestra para confirmar."
        )
    else:
        line("- ❌ El límite inferior es negativo → edge no demostrado aún.")
    line()

    # ── 4. Calibración ───────────────────────────────────────────────────────
    h(2, "4. Calibración del Modelo (Reliability Diagram)")
    line(
        "> **Nota**: solo vemos apuestas con edge ≥ 5% → las probabilidades "
        "bajas (<0.55) no aparecen. Sesgo de selección esperado."
    )
    line()
    line("| Bucket p_modelo | n | p_esperada | p_observada | Δ | Calibrado? |")
    line("|-----------------|---|-----------|------------|---|------------|")
    for row in cal:
        line(
            f"| {row['bucket']} | {row['n']} | {_pct(row['expected_p'])} "
            f"| {_pct(row['observed_p'])} | {row['diff']:+.3f} | {row['calib']} |"
        )
    line()
    line(
        f"**Brier Score: {bs:.4f}**  (referencia aleatoria ≈ 0.25 · referencia perfecta = 0.00)"
    )
    line()
    if bs < 0.22:
        line("✅ Modelo bien calibrado.")
    elif bs < 0.25:
        line("⚠️  Calibración aceptable.")
    else:
        line("❌ Modelo sobreconfiado o mal calibrado.")
    line()

    # ── 5. Análisis por segmento ─────────────────────────────────────────────
    h(2, "5. Análisis por Segmento")

    def _seg_table(title: str, segs: list[tuple[str, dict]], min_n: int = 3) -> None:
        h(3, title)
        line("| Segmento | n | Hit% | ROI | PnL | Sharpe | Sig |")
        line("|----------|---|------|-----|-----|--------|-----|")
        for k, sm in segs:
            if sm.get("n", 0) < min_n:
                continue
            line(
                f"| {k} | {sm['n']} | {_pct(sm['hit_rate'])} "
                f"| {_pct(sm['roi'], 2)} | {sm['total_pnl']:+.2f}u "
                f"| {_ff(sm['sharpe'])} | {_sig(sm['p_value'])} |"
            )
        line()

    _seg_table("5.1 Por Fuente", segment_analysis(bets, "fuente"))
    _seg_table("5.2 Por Mercado", segment_analysis(bets, "mercado"))
    _seg_table("5.3 Por Lado (over/under)", segment_analysis(bets, "bet"))
    _seg_table(
        "5.4 Por Posición (local/visitante/total)", segment_analysis(bets, "side_label")
    )
    _seg_table("5.5 Por Bucket de Edge", edge_bucket_analysis(bets), min_n=2)

    # Equipos
    teams = sorted(
        team_analysis(bets, min_n=5), key=lambda x: x[1]["roi"], reverse=True
    )
    if teams:
        _seg_table("5.6 Top Equipos por ROI (min 5 apuestas)", teams[:12])
        worst = sorted(teams, key=lambda x: x[1]["roi"])[:8]
        _seg_table("5.7 Peores Equipos por ROI", worst)

    # Jornada
    h(3, "5.8 Por Jornada (curva PnL acumulado)")
    line("| Jornada | n | Hit% | PnL jornada | PnL acumulado |")
    line("|---------|---|------|-------------|---------------|")
    cumulative_pnl = 0.0
    for j, jm in jornada_analysis(bets):
        cumulative_pnl += jm["total_pnl"]
        sign = "📈" if cumulative_pnl >= 0 else "📉"
        line(
            f"| J{j} | {jm['n']} | {_pct(jm['hit_rate'])} "
            f"| {jm['total_pnl']:+.2f}u | {cumulative_pnl:+.2f}u {sign} |"
        )
    line()

    # ── 6. Distribución de edges ─────────────────────────────────────────────
    h(2, "6. Distribución de Edges")
    edge_segs = edge_bucket_analysis(bets)
    total_n = sum(sm["n"] for _, sm in edge_segs)
    for lbl, sm in edge_segs:
        pct_n = sm["n"] / total_n if total_n else 0
        bar = "█" * max(1, int(pct_n * 35))
        line(f"`{lbl}` {sm['n']:4d} bets ({_pct(pct_n)})  {bar}")
        line()

    # ── 7. Riesgos y advertencias ────────────────────────────────────────────
    h(2, "7. Riesgos y Advertencias")
    if m["n"] < 50:
        line(
            f"- ⚠️ **Muestra muy pequeña** ({m['n']} bets). Resultados indicativos, no concluyentes."
        )
    fuentes = {b["fuente"] for b in bets}
    if "CONSENSUS_CODERE_STYLE" in fuentes:
        line(
            "- ⚠️ **Cuotas sintéticas presentes** (CONSENSUS_CODERE_STYLE). "
            "El modelo fue calibrado para ajustarse al mercado → posible sobreajuste en este segmento."
        )
    if len(fuentes) > 1:
        line(
            f"- ℹ️ Dataset mixto: {', '.join(sorted(fuentes))}. "
            "Interpretar segmento CODERE como benchmark primario."
        )
    line(
        "- ℹ️ **Selección bias en calibración**: solo hay apuestas con edge ≥ 5%, "
        "por lo que el reliability diagram solo cubre la cola superior de p_model."
    )
    line(
        "- ℹ️ **Correlación intra-partido**: hasta 3 apuestas por fixture (total + local + visitante). "
        "Los tests estadísticos asumen independencia → ligeramente optimistas."
    )
    line()

    # ── 8. Conclusiones ──────────────────────────────────────────────────────
    h(2, "8. Conclusiones y Próximos Pasos")

    conclusions = []
    if m["roi"] > 0.15:
        conclusions.append(f"✅ ROI {_pct(m['roi'], 2)} → edge claro sobre el mercado.")
    elif m["roi"] > 0.05:
        conclusions.append(f"✅ ROI {_pct(m['roi'], 2)} → edge positivo.")
    elif m["roi"] > 0:
        conclusions.append(f"⚠️  ROI {_pct(m['roi'], 2)} → edge marginal.")
    else:
        conclusions.append(f"❌ ROI {_pct(m['roi'], 2)} → sin edge.")

    if m["p_value"] < 0.05:
        conclusions.append(f"✅ Significancia al 5% (p={m['p_value']:.4f}).")
    elif m["p_value"] < 0.10:
        conclusions.append(f"⚠️  Significancia marginal al 10% (p={m['p_value']:.4f}).")
    else:
        conclusions.append(
            f"⚠️  Sin significancia estadística (p={m['p_value']:.4f}) — "
            f"necesita ≈ {_min_n_for_sig(m):.0f} apuestas para confirmar."
        )

    if ci_lo > 0:
        conclusions.append(
            "✅ IC bootstrapped positivo en su límite inferior → edge sostenido."
        )
    elif ci_lo > -0.03:
        conclusions.append("⚠️  IC bootstrapped roza cero — ampliar muestra.")
    else:
        conclusions.append("❌ IC bootstrapped negativo — edge no probado.")

    best_seg = max(
        segment_analysis(bets, "mercado"),
        key=lambda x: x[1].get("roi", -999) if x[1].get("n", 0) >= 5 else -999,
    )
    if best_seg[1].get("n", 0) >= 5:
        conclusions.append(
            f"🏆 Mejor segmento: **{best_seg[0]}** "
            f"(ROI {_pct(best_seg[1]['roi'], 2)}, n={best_seg[1]['n']})."
        )

    for c in conclusions:
        line(f"- {c}")
    line()
    line("**Siguientes pasos recomendados:**")
    line()
    line("1. Ejecutar `staking_sim.py` para encontrar la fracción de Kelly óptima.")
    line(
        "2. Ampliar muestra: re-ejecutar `backtest_odds.py --source ALL` con Supabase activo."
    )
    line(
        "3. Integrar overlay narrativo en el backtest para comparar pre vs post overlay."
    )
    line("4. Si ROI CODERE se mantiene >10% con n>80 → verde para producción.")
    line()

    return "".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", type=Path, help="CSV(s) de backtest a analizar")
    ap.add_argument(
        "--label",
        default="",
        help="Etiqueta para el reporte (default: nombre del fichero)",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--bootstrap-n", type=int, default=10_000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"[WARN] No existe: {csv_path}", file=sys.stderr)
            continue

        label = args.label or csv_path.stem
        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"[INFO] Analizando: {csv_path}  (label={label})", file=sys.stderr)

        bets = load_csv(csv_path)
        if not bets:
            print(f"[WARN] Sin apuestas en {csv_path}", file=sys.stderr)
            continue

        print(f"[INFO] {len(bets)} apuestas cargadas.", file=sys.stderr)
        print("[INFO] Calculando bootstrap CI (puede tardar ~5s)...", file=sys.stderr)

        ci_lo, ci_hi = bootstrap_roi_ci(bets, n_resamples=args.bootstrap_n)

        report = generate_report(bets, label, ci_lo, ci_hi, source_path=str(csv_path))

        out_path = args.out_dir / f"analysis_{label}.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"[INFO] Reporte guardado: {out_path}", file=sys.stderr)
        print(report)


if __name__ == "__main__":
    main()
