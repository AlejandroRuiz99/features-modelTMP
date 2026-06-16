#!/usr/bin/env python3
"""
staking_sim.py — Simulación Monte Carlo de bankroll con sweep de fracciones de Kelly.

Para cada fracción de Kelly evaluada:
  1. Usa la distribución empírica de apuestas del backtest como base
  2. Simula N_PATHS caminos de bankroll (remuestreo con reemplazo)
  3. Calcula: bankroll terminal (mediana, p5, p95), P(ruina), max drawdown,
     Sharpe del camino, ROI mediano
  4. Genera tabla comparativa + recomendación de fracción óptima

Salida:
  - reports/staking_sim_{label}.md   (reporte Markdown)
  - reports/staking_sim_{label}.csv  (datos por fracción)

Uso:
    python scripts/staking_sim.py reports/backtest_codere_e05.csv
    python scripts/staking_sim.py reports/backtest_all_e05.csv --label all
    python scripts/staking_sim.py reports/backtest_codere_e05.csv --fractions 0.1 0.25 0.5 1.0 2.0
    python scripts/staking_sim.py reports/backtest_codere_e05.csv --paths 20000 --bets-per-path 200
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# CSV loader (igual que analyze_backtest.py — standalone)
# ---------------------------------------------------------------------------


def _f(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _detect_delimiter(path: Path) -> str:
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline()
    return ";" if first.count(";") > first.count(",") else ","


def load_csv(path: Path) -> list[dict]:
    delim = _detect_delimiter(path)
    bets = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv_mod.DictReader(f, delimiter=delim)
        for r in reader:
            edge = _f(r.get("edge"))
            odds = _f(r.get("odds"))
            won_raw = r.get("won")
            pnl = _f(r.get("pnl"))
            if edge is None or odds is None or won_raw is None:
                continue
            won = int(float(won_raw))
            if pnl is None:
                pnl = (odds - 1.0) if won else -1.0
            bets.append(
                {
                    "edge": edge,
                    "odds": odds,
                    "won": won,
                    "pnl": pnl,
                    "mercado": (r.get("mercado") or "").strip(),
                    "fuente": (r.get("fuente") or "").strip(),
                    "bet": (r.get("bet") or "").strip(),
                }
            )
    return bets


# ---------------------------------------------------------------------------
# Kelly helpers
# ---------------------------------------------------------------------------


def full_kelly(edge: float, odds: float) -> float:
    """Fracción de Kelly para apuesta binaria.

    f* = (p·b - q) / b  donde b = odds-1, p = prob win, q = 1-p.
    p se estima como (1 + edge) / odds  (la prob implícita por el edge).
    """
    if odds <= 1.0:
        return 0.0
    p = (1.0 + edge) / odds
    p = min(max(p, 0.0), 1.0)
    q = 1.0 - p
    b = odds - 1.0
    f = (p * b - q) / b
    return max(0.0, f)


# ---------------------------------------------------------------------------
# Monte Carlo core
# ---------------------------------------------------------------------------

_RUIN_THRESHOLD = 0.40  # bankroll < 40% del inicial → ruina


def simulate_paths(
    bets: list[dict],
    fraction: float,
    n_paths: int,
    bets_per_path: int,
    bankroll_init: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    # Vectores base: kelly stake completo, won (bool), net odds
    n_emp = len(bets)
    fk_arr = np.array([full_kelly(b["edge"], b["odds"]) for b in bets])
    won_arr = np.array([b["won"] for b in bets], dtype=float)
    net_arr = np.array([b["odds"] - 1.0 for b in bets])

    terminal = np.empty(n_paths)
    max_dd_arr = np.empty(n_paths)
    ruin_count = 0
    all_sharpes = []

    for trial in range(n_paths):
        idx = rng.integers(0, n_emp, size=bets_per_path)
        fk = fk_arr[idx]
        won = won_arr[idx]
        net = net_arr[idx]

        bankroll = bankroll_init
        peak = bankroll_init
        max_dd = 0.0
        ruined = False
        path_pnl = []

        for i in range(bets_per_path):
            stake_frac = fraction * fk[i]
            stake_frac = min(stake_frac, 0.99)  # no apostar más del 99% del bankroll
            stake = stake_frac * bankroll

            if won[i]:
                delta = stake * net[i]
            else:
                delta = -stake

            bankroll += delta
            path_pnl.append(delta)

            if bankroll > peak:
                peak = bankroll
            if peak > 0:
                dd = (peak - bankroll) / peak
                if dd > max_dd:
                    max_dd = dd

            if bankroll < _RUIN_THRESHOLD * bankroll_init:
                ruined = True
                break

        if ruined:
            ruin_count += 1

        terminal[trial] = max(0.0, bankroll)
        max_dd_arr[trial] = max_dd

        # Sharpe del camino
        if len(path_pnl) > 1:
            pnl_arr = np.array(path_pnl)
            std = float(np.std(pnl_arr, ddof=1))
            sharpe = float(np.mean(pnl_arr) / std) if std > 1e-9 else 0.0
        else:
            sharpe = 0.0
        all_sharpes.append(sharpe)

    roi_arr = terminal / bankroll_init - 1.0
    sharpes_arr = np.array(all_sharpes)

    return {
        "fraction": fraction,
        "n_paths": n_paths,
        "bets_per_path": bets_per_path,
        # Bankroll terminal
        "median_final": float(np.median(terminal)),
        "mean_final": float(np.mean(terminal)),
        "p5_final": float(np.percentile(terminal, 5)),
        "p25_final": float(np.percentile(terminal, 25)),
        "p75_final": float(np.percentile(terminal, 75)),
        "p95_final": float(np.percentile(terminal, 95)),
        # ROI
        "roi_median": float(np.median(roi_arr)),
        "roi_mean": float(np.mean(roi_arr)),
        "roi_p5": float(np.percentile(roi_arr, 5)),
        "roi_p95": float(np.percentile(roi_arr, 95)),
        # Riesgo
        "p_ruin": ruin_count / n_paths,
        "median_max_dd": float(np.median(max_dd_arr)),
        "p95_max_dd": float(np.percentile(max_dd_arr, 95)),
        # Sharpe
        "median_sharpe": float(np.median(sharpes_arr)),
        # Score compuesto: penaliza P(ruin) y max drawdown alto
        "score": _composite_score(
            float(np.median(roi_arr)),
            ruin_count / n_paths,
            float(np.percentile(max_dd_arr, 95)),
        ),
    }


def _composite_score(roi_median: float, p_ruin: float, dd_p95: float) -> float:
    """Score [0-100]: combina ROI mediano, P(ruin) y drawdown P95.

    No hay fórmula única 'correcta' — esto prioriza:
      - Maximizar ROI mediano
      - Minimizar P(ruin) (penalización fuerte)
      - Minimizar drawdown P95 (penalización moderada)
    """
    roi_norm = min(max(roi_median, 0.0), 2.0) / 2.0  # [0, 1]
    ruin_pen = 1.0 - min(p_ruin * 3.0, 1.0)  # 0 si P(ruin)>33%
    dd_pen = 1.0 - min(dd_p95 * 1.5, 1.0)  # 0 si DD>67%
    return round(100.0 * roi_norm * ruin_pen * dd_pen, 1)


# ---------------------------------------------------------------------------
# Análisis por segmento de mercado
# ---------------------------------------------------------------------------


def segment_sim(
    bets: list[dict],
    fraction: float,
    n_paths: int,
    bets_per_path: int,
    seed: int,
) -> list[tuple[str, dict]]:
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for b in bets:
        groups[b["mercado"]].append(b)

    results = []
    for mkt, group in sorted(groups.items()):
        if len(group) < 5:
            continue
        bpp = min(bets_per_path, max(10, len(group)))
        r = simulate_paths(
            group, fraction, n_paths, bpp, 100.0, seed + hash(mkt) % 9999
        )
        r["mercado"] = mkt
        results.append((mkt, r))
    return results


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%"


def _ff(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def generate_report(
    results: list[dict],
    seg_results: list[tuple[str, dict]],
    label: str,
    bankroll_init: float,
    bets_per_path: int,
    n_paths: int,
    n_empirical: int,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L: list[str] = []

    def line(s: str = "") -> None:
        L.append(s + "\n")

    def h(n: int, t: str) -> None:
        line(f"{'#' * n} {t}")
        line()

    h(1, f"Simulación Monte Carlo — foultsPredictor · `{label}`")
    line(f"_Generado: {ts}_")
    line()
    line(f"- Apuestas empíricas: **{n_empirical}**")
    line(f"- Bankroll inicial normalizado: **{bankroll_init:.0f}u**")
    line(f"- Apuestas por camino: **{bets_per_path}**")
    line(f"- Caminos por fracción: **{n_paths:,}**")
    line(f"- Umbral de ruina: **<{_pct(_RUIN_THRESHOLD)} del bankroll inicial**")
    line()

    # Tabla principal
    h(2, "1. Comparativa de Fracciones de Kelly")
    line(
        "| Fracción Kelly | ROI mediano | ROI p5 | P(ruina) | MaxDD p95 | Bankroll mediano | Score |"
    )
    line(
        "|----------------|-------------|--------|----------|-----------|-----------------|-------|"
    )
    best = max(results, key=lambda r: r["score"])
    for r in results:
        flag = " ← **ÓPTIMO**" if r["fraction"] == best["fraction"] else ""
        line(
            f"| {r['fraction']:.2f}x | {_pct(r['roi_median'], 1)} "
            f"| {_pct(r['roi_p5'], 1)} | {_pct(r['p_ruin'], 1)} "
            f"| {_pct(r['median_max_dd'], 1)} | {r['median_final']:.1f}u "
            f"| {r['score']:.0f}/100 |{flag}"
        )
    line()

    # Detalle fracción óptima
    h(2, f"2. Detalle Fracción Óptima: {best['fraction']:.2f}x Kelly")
    line("| Métrica | Valor |")
    line("|---------|-------|")
    detail_rows = [
        ("Fracción de Kelly aplicada", f"{best['fraction']:.2f}x"),
        ("ROI mediano", _pct(best["roi_median"], 2)),
        ("ROI media", _pct(best["roi_mean"], 2)),
        ("ROI p5 (escenario adverso)", _pct(best["roi_p5"], 2)),
        ("ROI p95 (escenario favorable)", _pct(best["roi_p95"], 2)),
        (
            "Bankroll terminal mediano",
            f"{best['median_final']:.1f}u (+{_pct(best['roi_median'], 1)})",
        ),
        ("Bankroll terminal p5", f"{best['p5_final']:.1f}u"),
        ("Bankroll terminal p95", f"{best['p95_final']:.1f}u"),
        ("P(ruina)", _pct(best["p_ruin"], 2)),
        ("Max Drawdown mediano", _pct(best["median_max_dd"], 1)),
        ("Max Drawdown p95", _pct(best["p95_max_dd"], 1)),
        ("Sharpe mediano (per-bet)", _ff(best["median_sharpe"])),
        ("Score compuesto", f"{best['score']:.0f}/100"),
    ]
    for name, val in detail_rows:
        line(f"| {name} | {val} |")
    line()

    # Análisis de riesgo por fracción
    h(2, "3. Análisis de Riesgo")
    line("```")
    line(
        f"{'Fracción':>10} | {'P(ruina)':>10} | {'MaxDD p95':>10} | {'Bankroll p5':>12}"
    )
    line("-" * 52)
    for r in results:
        line(
            f"{r['fraction']:>10.2f}x | {_pct(r['p_ruin'], 1):>10} "
            f"| {_pct(r['p95_max_dd'], 1):>10} | {r['p5_final']:>10.1f}u"
        )
    line("```")
    line()

    # Distribución bankroll terminal — histograma ASCII por fracción
    h(2, "4. Distribución Bankroll Terminal (percentiles)")
    line("| Fracción | p5 | p25 | Mediana | p75 | p95 |")
    line("|----------|----|-----|---------|-----|-----|")
    for r in results:
        line(
            f"| {r['fraction']:.2f}x "
            f"| {r['p5_final']:.0f}u "
            f"| {r['p25_final']:.0f}u "
            f"| {r['median_final']:.0f}u "
            f"| {r['p75_final']:.0f}u "
            f"| {r['p95_final']:.0f}u |"
        )
    line()

    # Segmento por mercado
    if seg_results:
        h(2, "5. Fracción Óptima por Mercado")
        line(
            "> Simulado con la fracción óptima global. "
            "Útil para ajustar kelly_scale por tipo de mercado."
        )
        line()
        line("| Mercado | n bets | ROI mediano | P(ruina) | MaxDD p95 | Score |")
        line("|---------|--------|-------------|----------|-----------|-------|")
        for mkt, r in sorted(seg_results, key=lambda x: x[1]["score"], reverse=True):
            line(
                f"| {mkt} | {n_empirical} | {_pct(r['roi_median'], 1)} "
                f"| {_pct(r['p_ruin'], 1)} | {_pct(r['p95_max_dd'], 1)} "
                f"| {r['score']:.0f}/100 |"
            )
        line()

    # Recomendaciones
    h(2, "6. Recomendaciones")
    bf = best["fraction"]
    line(f"**Fracción de Kelly recomendada: {bf:.2f}x**")
    line()

    if bf <= 0.25:
        line(
            "- Fracción conservadora. Adecuada dado el tamaño de muestra. "
            "Revisar cuando n > 100 en CODERE real."
        )
    elif bf <= 0.5:
        line(
            "- Fracción moderada. Buen balance entre crecimiento y control del riesgo."
        )
    else:
        line(
            "- Fracción agresiva. Requiere alta confianza en la estimación del edge. "
            "Considerar fracción más conservadora mientras la muestra sea < 100."
        )

    if best["p_ruin"] > 0.05:
        line(
            f"- ⚠️  P(ruina) del {_pct(best['p_ruin'], 1)} con la fracción óptima. "
            "Considera reducir o aplicar stop-loss."
        )
    else:
        line(f"- ✅ P(ruina) del {_pct(best['p_ruin'], 1)} — riesgo controlado.")

    if best["p95_max_dd"] > 0.35:
        line(
            f"- ⚠️  Drawdown máximo (p95) del {_pct(best['p95_max_dd'], 1)}. "
            "Implementar stop-loss del 20-25%."
        )

    line()
    line(
        "**Traducción a staking.yaml**: multiplica `bankroll_share_per_stake_unit` "
        f"por {bf:.2f} para alinear el sistema de staking con la fracción óptima."
    )
    line()

    # Advertencias
    h(2, "7. Advertencias Metodológicas")
    line(
        f"- La distribución empírica tiene **{n_empirical} apuestas** — "
        "remuestreo con reemplazo puede sobreestimar consistencia del edge."
    )
    line(
        "- La función `full_kelly` asume que el edge estimado en el backtest "
        "es igual al edge real futuro. En la práctica, aplica un descuento adicional."
    )
    line(
        "- Las apuestas del mismo partido están correlacionadas. "
        "El simulador las trata como independientes → P(ruina) ligeramente subestimada."
    )
    line(
        "- Con muestra < 50 (especialmente CODERE), los resultados son orientativos. "
        "Prioriza fracciones bajas (0.10x–0.25x) hasta ampliar la muestra."
    )
    line()

    return "".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_FRACTIONS = [0.10, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csv", type=Path, help="CSV de backtest (generado por backtest_odds.py)"
    )
    ap.add_argument("--label", default="", help="Etiqueta para el reporte")
    ap.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=DEFAULT_FRACTIONS,
        help="Fracciones de Kelly a evaluar",
    )
    ap.add_argument(
        "--paths", type=int, default=10_000, help="Número de caminos Monte Carlo"
    )
    ap.add_argument(
        "--bets-per-path", type=int, default=150, help="Apuestas por camino"
    )
    ap.add_argument(
        "--bankroll", type=float, default=100.0, help="Bankroll inicial normalizado"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[ERROR] No existe: {args.csv}", file=sys.stderr)
        sys.exit(1)

    label = args.label or args.csv.stem
    bets = load_csv(args.csv)
    if not bets:
        print("[ERROR] Sin apuestas cargadas.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(bets)} apuestas empíricas cargadas.", file=sys.stderr)
    print(
        f"[INFO] Simulando {len(args.fractions)} fracciones × {args.paths:,} caminos × "
        f"{args.bets_per_path} apuestas/camino...",
        file=sys.stderr,
    )

    results = []
    for i, frac in enumerate(sorted(args.fractions)):
        print(
            f"  [{i + 1}/{len(args.fractions)}] fracción={frac:.2f}x...",
            end="\r",
            file=sys.stderr,
        )
        r = simulate_paths(
            bets,
            fraction=frac,
            n_paths=args.paths,
            bets_per_path=args.bets_per_path,
            bankroll_init=args.bankroll,
            seed=args.seed,
        )
        results.append(r)
    print("", file=sys.stderr)

    # Segmento por mercado con la fracción óptima
    best_frac = max(results, key=lambda r: r["score"])["fraction"]
    print(
        f"[INFO] Fracción óptima: {best_frac:.2f}x. Simulando por mercado...",
        file=sys.stderr,
    )
    seg_results = segment_sim(
        bets, best_frac, args.paths // 2, args.bets_per_path, args.seed
    )

    # Reporte
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = generate_report(
        results,
        seg_results,
        label=label,
        bankroll_init=args.bankroll,
        bets_per_path=args.bets_per_path,
        n_paths=args.paths,
        n_empirical=len(bets),
    )

    out_md = args.out_dir / f"staking_sim_{label}.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"[INFO] Reporte guardado: {out_md}", file=sys.stderr)

    # CSV de resultados
    out_csv = args.out_dir / f"staking_sim_{label}.csv"
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"[INFO] Datos CSV guardados: {out_csv}", file=sys.stderr)

    print(report)


if __name__ == "__main__":
    main()
