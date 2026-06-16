#!/usr/bin/env python3
"""
run_data_miner.py — Compara predicciones CON contexto (runs/) contra el
backtest automático SIN árbitro ni overlay.

Responde: ¿el rendimiento de J28-J38 se explica por árbitro + overlay?
"""

import csv
import json
import math
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
BACKTEST_CSV = ROOT / "reports" / "backtest_all_e05.csv"
OUTPUT_CSV = ROOT / "reports" / "context_comparison.csv"
OUTPUT_MD = ROOT / "reports" / "context_vs_backtest.md"


# ── helpers ──────────────────────────────────────────────────────────────────


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def norm(name: str) -> str:
    n = name.lower().strip()
    for tok in [
        " cf",
        " fc",
        " s.a.d.",
        "real ",
        "club ",
        "atlético",
        "atletico",
        "athletic",
        "deportivo",
        "rayo",
        " b",
        "(b)",
    ]:
        n = n.replace(tok, "").strip()
    return n


def poisson_p_over(mu: float, line: float) -> float:
    """P(X > line) = P(X >= floor(line)+1) using Poisson PMF directly."""
    k = int(line)  # floor
    if mu <= 0:
        return 0.0
    # compute CDF(k) = sum_{i=0}^{k} e^{-mu} * mu^i / i!
    log_mu = math.log(mu)
    cdf = 0.0
    log_p = -mu  # log P(X=0)
    cdf += math.exp(log_p)
    for i in range(1, k + 1):
        log_p += log_mu - math.log(i)
        cdf += math.exp(log_p)
        if math.exp(log_p) < 1e-12:
            break
    return max(0.0, min(1.0, 1.0 - cdf))


MERCADO_MAP = {
    "total fouls": "total",
    "team total fouls local": "local",
    "team total fouls visitante": "visitante",
    "home team total fouls": "local",
    "away team total fouls": "visitante",
}


def norm_mercado(raw: str) -> str:
    """Map raw mercado string to 'total'/'local'/'visitante'."""
    return MERCADO_MAP.get(raw.lower().strip(), raw.lower().strip())


def norm_date(raw: str) -> str:
    """Convert DD/MM/YYYY or YYYY-MM-DD to YYYY-MM-DD."""
    raw = raw.strip()
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            d, m, y = parts
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    return raw


def load_backtest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        sample = f.read(2000)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=delim))
    # Normalize dates and mercado
    for r in rows:
        r["date"] = norm_date(r.get("date", ""))
        r["_mercado_norm"] = norm_mercado(r.get("mercado", ""))
        r["_bet_dir"] = r.get("bet", "").lower().strip()  # 'over' or 'under'
    return rows


# ── run miner ────────────────────────────────────────────────────────────────


def mine_runs(runs_dir: Path) -> list[dict]:
    records = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        pred_file = run_dir / "prediction" / "prediction.json"
        if not pred_file.exists():
            continue
        try:
            preds = json.loads(pred_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN {pred_file}: {e}", file=sys.stderr)
            continue

        for p in preds:
            match_str = p.get("match", "")
            parts = match_str.split(" vs ", 1)
            if len(parts) != 2:
                continue
            home_r, away_r = parts[0].strip(), parts[1].strip()
            ov = p.get("overlay", {})
            records.append(
                {
                    "run_id": run_dir.name,
                    "date": p.get("date", ""),
                    "jornada": p.get("jornada"),
                    "home_run": home_r,
                    "away_run": away_r,
                    "referee": p.get("referee", ""),
                    "referee_strict_prob": p.get("referee_strict_prob", 0.5),
                    "home_expected": p.get("home_expected"),
                    "away_expected": p.get("away_expected"),
                    "total_expected": p.get("total_expected")
                    or p.get("expected_fouls"),
                    "pre_expected": ov.get("pre_expected_fouls"),
                    "post_expected": ov.get("post_expected_fouls"),
                    "delta_fouls": ov.get("delta_fouls_applied", 0),
                    "variance_scale": ov.get("variance_scale_applied", 1.0),
                    "kelly_scale": ov.get("kelly_scale_applied", 1.0),
                    "rules_fired": "|".join(
                        r.get("id", "") for r in ov.get("rules_fired", [])
                    ),
                    "over_under": p.get("over_under", {}),
                }
            )
    return records


def get_p_over_run(rec: dict, mercado: str, linea: float) -> float | None:
    """
    p_over from run prediction for a given market and line.
    total → use over_under dict (already has overlay-adjusted PMF).
    local/visitante → Poisson from home/away_expected (pre-overlay approx).
    """
    if mercado == "total":
        ou = rec["over_under"]
        key = str(linea)
        if key in ou:
            return float(ou[key]["over"])
        for k, v in ou.items():
            if abs(float(k) - linea) < 0.01:
                return float(v["over"])
        return None
    else:
        mu = rec["home_expected"] if mercado == "local" else rec["away_expected"]
        if mu is None:
            return None
        return poisson_p_over(float(mu), linea)


# ── matching ─────────────────────────────────────────────────────────────────


def match_bets(backtest_rows: list[dict], run_records: list[dict]) -> list[dict]:
    # Index by (date, norm_home, norm_away)
    exact_idx: dict[tuple, list[dict]] = defaultdict(list)
    date_idx: dict[str, list[dict]] = defaultdict(list)
    for r in run_records:
        exact_idx[(r["date"], norm(r["home_run"]), norm(r["away_run"]))].append(r)
        date_idx[r["date"]].append(r)

    results = []
    unmatched = 0

    for row in backtest_rows:
        jornada = int(float(row.get("jornada", 0)))
        if jornada < 28:
            continue

        date = row.get("date", "")  # already normalized to YYYY-MM-DD
        home_bt = row.get("home", "")
        away_bt = row.get("away", "")
        mercado = row.get("_mercado_norm", "")  # 'total'/'local'/'visitante'
        side = row.get("_bet_dir", "over")  # 'over' or 'under'
        linea = float(row.get("linea", 0))
        odds = float(row.get("odds", 1.0))
        p_over_bt = float(row.get("p_over_model", 0))

        # try exact match
        matched_rec = None
        key = (date, norm(home_bt), norm(away_bt))
        if key in exact_idx:
            matched_rec = exact_idx[key][0]
        else:
            # fuzzy on same date
            best_score = 0.0
            for cand in date_idx.get(date, []):
                score = (
                    sim(home_bt, cand["home_run"]) + sim(away_bt, cand["away_run"])
                ) / 2
                if score > best_score:
                    best_score = score
                    matched_rec = cand
            if best_score < 0.50:
                matched_rec = None

        if matched_rec is None:
            unmatched += 1
            continue

        p_over_run = get_p_over_run(matched_rec, mercado, linea)

        p_bet_bt = p_over_bt if side == "over" else (1 - p_over_bt)
        edge_bt = p_bet_bt - 1 / odds

        p_bet_run = None
        edge_run = None
        bet_run = None
        if p_over_run is not None:
            p_bet_run = p_over_run if side == "over" else (1 - p_over_run)
            edge_run = p_bet_run - 1 / odds
            bet_run = edge_run >= 0.05

        delta_p_over = (p_over_run - p_over_bt) if p_over_run is not None else None

        # delta_mu: how much overlay moved total μ
        pre = matched_rec.get("pre_expected")
        post = matched_rec.get("post_expected")
        delta_mu = (post - pre) if (pre and post) else matched_rec.get("delta_fouls", 0)

        results.append(
            {
                **row,
                "run_id": matched_rec["run_id"],
                "referee": matched_rec["referee"],
                "referee_strict_prob": matched_rec["referee_strict_prob"],
                "pre_expected": pre,
                "post_expected": post,
                "delta_mu": delta_mu,
                "delta_fouls": matched_rec["delta_fouls"],
                "variance_scale": matched_rec["variance_scale"],
                "kelly_scale": matched_rec["kelly_scale"],
                "rules_fired": matched_rec["rules_fired"],
                "p_over_run": round(p_over_run, 6) if p_over_run is not None else "",
                "p_bet_run": round(p_bet_run, 6) if p_bet_run is not None else "",
                "delta_p_over": round(delta_p_over, 6)
                if delta_p_over is not None
                else "",
                "edge_run": round(edge_run, 6) if edge_run is not None else "",
                "bet_run": bet_run,
            }
        )

    if unmatched:
        print(f"  WARN: {unmatched} bets (J28+) sin match en runs/", file=sys.stderr)
    return results


# ── analysis ─────────────────────────────────────────────────────────────────


def _is_bet(r: dict) -> bool:
    """True if this row represents a placed bet (bet_dir = 'over' or 'under')."""
    return str(r.get("_bet_dir", r.get("bet", ""))).lower().strip() in ("over", "under")


def _is_won(r: dict) -> bool:
    return str(r.get("won", "")).strip() in ("1", "True", "true")


def _hr(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if _is_won(r)) / len(rows)


def _roi(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    total_pnl = sum(float(r.get("pnl", 0)) for r in rows)
    return total_pnl / len(rows) * 100


def _fmt(val, fmt=".3f") -> str:
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "N/A"


def analyze(matched: list[dict]) -> str:
    L = []
    n = len(matched)

    bets_bt = [r for r in matched if _is_bet(r)]
    n_bt = len(bets_bt)

    has_dp = [r for r in matched if r.get("delta_p_over") not in ["", None]]
    delta_ps = [float(r["delta_p_over"]) for r in has_dp]
    delta_mus = [
        float(r["delta_mu"]) for r in matched if r.get("delta_mu") not in ["", None]
    ]

    ctx_up = [r for r in has_dp if float(r["delta_p_over"]) > 0.02]
    ctx_down = [r for r in has_dp if float(r["delta_p_over"]) < -0.02]
    ctx_flat = [r for r in has_dp if abs(float(r["delta_p_over"])) <= 0.02]

    both = [r for r in matched if _is_bet(r) and r.get("bet_run") is True]
    bt_only = [r for r in matched if _is_bet(r) and r.get("bet_run") is False]
    run_only = [r for r in matched if not _is_bet(r) and r.get("bet_run") is True]

    bt_up = [
        r
        for r in bets_bt
        if r.get("delta_p_over") not in ["", None] and float(r["delta_p_over"]) > 0.02
    ]
    bt_down = [
        r
        for r in bets_bt
        if r.get("delta_p_over") not in ["", None] and float(r["delta_p_over"]) < -0.02
    ]
    bt_flat = [
        r
        for r in bets_bt
        if r.get("delta_p_over") not in ["", None]
        and abs(float(r["delta_p_over"])) <= 0.02
    ]

    L.append(
        "# EXP-5: Contexto vs Backtest — ¿Árbitro + Overlay Explican el Rendimiento?\n"
    )
    L.append(
        f"> **{n}** apuestas de J28-J38 matcheadas con runs/ | "
        f"**{n_bt}** apostadas por el backtest | "
        f"**{len(has_dp)}** con delta_p calculado\n"
    )

    L.append("---\n")

    # ── Section 1: Magnitude
    L.append("## 1. Magnitud del Impacto del Contexto\n")
    if delta_ps:
        avg_dp = sum(delta_ps) / len(delta_ps)
        abs_dp = sum(abs(x) for x in delta_ps) / len(delta_ps)
        max_up = max(delta_ps)
        max_dn = min(delta_ps)
        avg_dmu = sum(delta_mus) / len(delta_mus) if delta_mus else 0
        pct_up = len(ctx_up) / len(has_dp) * 100
        pct_down = len(ctx_down) / len(has_dp) * 100
        pct_flat = len(ctx_flat) / len(has_dp) * 100

        L.append("| Métrica | Valor |")
        L.append("|---------|-------|")
        L.append(f"| Δp_over medio (sesgo) | {avg_dp:+.4f} |")
        L.append(f"| |Δp_over| medio (magnitud) | {abs_dp:.4f} |")
        L.append(f"| Δp_over MAX subida | {max_up:+.4f} |")
        L.append(f"| Δp_over MAX bajada | {max_dn:+.4f} |")
        L.append(f"| Δμ fouls medio (overlay) | {avg_dmu:+.4f} |")
        L.append(f"| Contexto sube >2pp | {len(ctx_up)} ({pct_up:.0f}%) |")
        L.append(f"| Contexto baja >2pp | {len(ctx_down)} ({pct_down:.0f}%) |")
        L.append(f"| Sin cambio (<= 2pp) | {len(ctx_flat)} ({pct_flat:.0f}%) |")
    else:
        L.append("*No hay datos de delta_p disponibles.*")
    L.append("")

    # ── Section 2: Decision changes
    L.append("## 2. Cambios de Decisión de Apuesta\n")
    L.append("| Categoría | N | Hit Rate | ROI |")
    L.append("|-----------|---|----------|-----|")
    L.append(
        f"| Ambos apuestan (BT + Context) | {len(both)} | {_hr(both):.1%} | {_roi(both):.1f}% |"
    )
    L.append(
        f"| Solo backtest (context cancela) | {len(bt_only)} | {_hr(bt_only):.1%} | {_roi(bt_only):.1f}% |"
    )
    L.append(
        f"| Solo context (backtest no entra) | {len(run_only)} | {_hr(run_only):.1%} | {_roi(run_only):.1f}% |"
    )
    L.append("")
    L.append(
        "> Si `solo_context` tiene buen ROI: el árbitro/overlay añade bets valiosos."
    )
    L.append(
        "> Si `bt_only` tiene mal ROI: el contexto filtra bien las apuestas dudosas."
    )
    L.append("")

    # ── Section 3: Does context predict outcome?
    L.append("## 3. ¿El Contexto Predice el Resultado?\n")
    L.append(
        "Entre las apuestas del backtest (J28-J38), ¿acertó el contexto al subir/bajar p?\n"
    )
    L.append("| Dirección Contexto | N | Ganadas | Hit Rate | ROI |")
    L.append("|--------------------|---|---------|----------|-----|")
    L.append(
        f"| Subió >+2pp → más confiado | {len(bt_up)} | {sum(1 for r in bt_up if _is_won(r))} | {_hr(bt_up):.1%} | {_roi(bt_up):.1f}% |"
    )
    L.append(
        f"| Bajó <-2pp → menos confiado | {len(bt_down)} | {sum(1 for r in bt_down if _is_won(r))} | {_hr(bt_down):.1%} | {_roi(bt_down):.1f}% |"
    )
    L.append(
        f"| Sin cambio significativo | {len(bt_flat)} | {sum(1 for r in bt_flat if _is_won(r))} | {_hr(bt_flat):.1%} | {_roi(bt_flat):.1f}% |"
    )
    L.append("")
    L.append(
        "> **Señal predictiva ideal**: hit_rate(UP) > hit_rate(FLAT) > hit_rate(DOWN)"
    )
    L.append("")

    # ── Section 4: Referee analysis
    L.append("## 4. Análisis por Árbitro\n")
    ref_groups: dict[str, list[dict]] = defaultdict(list)
    for r in matched:
        ref = r.get("referee", "")
        if ref:
            ref_groups[ref].append(r)

    L.append("| Árbitro | Partidos | Apuestas BT | Hit Rate | Δμ medio | Kelly Scale |")
    L.append(
        "|---------|----------|-------------|----------|-----------|-------------|"
    )
    for ref, rows in sorted(ref_groups.items(), key=lambda x: -len(x[1])):
        bets_ref = [r for r in rows if _is_bet(r)]
        n_matches = len(set((r.get("date", ""), r.get("home", "")) for r in rows))
        dmu = sum(float(r.get("delta_fouls", 0) or 0) for r in rows) / len(rows)
        ks = sum(float(r.get("kelly_scale", 1) or 1) for r in rows) / len(rows)
        L.append(
            f"| {ref} | {n_matches} | {len(bets_ref)} | {_hr(bets_ref):.1%} | {dmu:+.2f} | {ks:.2f} |"
        )
    L.append("")

    # ── Section 5: Overlay rules
    L.append("## 5. Distribución de Overlay Rules Activadas\n")
    rule_cnt: dict[str, int] = defaultdict(int)
    for r in matched:
        for rule in (r.get("rules_fired") or "").split("|"):
            if rule:
                rule_cnt[rule] += 1

    if rule_cnt:
        L.append("| Regla | Activaciones |")
        L.append("|-------|-------------|")
        for rule, cnt in sorted(rule_cnt.items(), key=lambda x: -x[1]):
            L.append(f"| {rule} | {cnt} |")
    else:
        L.append("*Ninguna regla registrada.*")
    L.append("")

    # ── Section 6: Top movers
    L.append("## 6. Top 10 Apuestas con Mayor Divergencia (|Δp| > 5pp)\n")
    big_movers = sorted(
        [
            r
            for r in matched
            if r.get("delta_p_over") not in ["", None]
            and abs(float(r["delta_p_over"])) > 0.05
        ],
        key=lambda x: -abs(float(x["delta_p_over"])),
    )[:10]

    if big_movers:
        L.append("| Fixture | Mercado | Línea | p_bt | p_run | Δp | Overlay | Won |")
        L.append("|---------|---------|-------|------|-------|----|---------|-----|")
        for r in big_movers:
            fix = f"{r.get('home', '')[:10]} vs {r.get('away', '')[:10]}"
            won = "✓" if _is_won(r) else "✗"
            dp = float(r["delta_p_over"])
            L.append(
                f"| {fix} | {r.get('mercado', '')} | {r.get('linea', '')} "
                f"| {_fmt(r.get('p_over_model'), '.3f')} "
                f"| {_fmt(r.get('p_over_run'), '.3f')} "
                f"| {dp:+.3f} "
                f"| {r.get('delta_fouls', '?')} fouls | {won} |"
            )
    else:
        L.append("*Sin divergencias >5pp encontradas.*")
    L.append("")

    # ── Section 7: Conclusion
    L.append("## 7. Conclusión\n")
    L.append("### Hipótesis original")
    L.append(
        "> *¿El mejor rendimiento en J28-J38 viene de haber usado árbitro + contexto?*\n"
    )

    if delta_ps:
        abs_dp_avg = sum(abs(x) for x in delta_ps) / len(delta_ps)
        avg_dmu_val = sum(delta_mus) / len(delta_mus) if delta_mus else 0

        L.append("**Evidencia cuantitativa**:")
        L.append(
            f"- El contexto mueve p_over una media de **{abs_dp_avg:.1%}** (magnitud absoluta)."
        )
        L.append(
            f"- El overlay ajusta μ una media de **{avg_dmu_val:+.2f} fouls** por partido."
        )

        if len(bt_up) >= 3 and len(bt_down) >= 3:
            if _hr(bt_up) > _hr(bt_flat) > _hr(bt_down):
                L.append(
                    "- **Señal predictiva confirmada**: bets donde contexto sube p ganan más; donde baja, menos."
                )
            elif _hr(bt_up) > _hr(bt_down):
                L.append(
                    "- **Señal parcial**: bets con contexto UP ganan más que DOWN, patrón flat inconsistente."
                )
            else:
                L.append(
                    "- **Sin señal clara**: el contexto no ordena bien los resultados (posible ruido por n pequeño)."
                )

        if len(bt_only) >= 3:
            if _roi(bt_only) < 0:
                L.append(
                    f"- **Filtrado valioso**: las {len(bt_only)} apuestas que el contexto habría cancelado tienen ROI "
                    f"{_roi(bt_only):.1f}% → el overlay descarta pérdidas."
                )
            else:
                L.append(
                    f"- Las apuestas que contexto cancela tienen ROI {_roi(bt_only):.1f}% → filtrado neutro."
                )

        L.append("")
        L.append("**Veredicto**:")
        if abs_dp_avg >= 0.04 and (len(bt_up) < 3 or _hr(bt_up) >= _hr(bt_down)):
            L.append(
                "El contexto tiene **impacto real** en las probabilidades. "
                "Parte del rendimiento de J28-J38 probablemente se debe al uso de árbitro + overlay. "
                "Para cuantificar con precisión se necesita un backtest con referee injection (EXP-6)."
            )
        elif abs_dp_avg >= 0.02:
            L.append(
                "El contexto tiene **impacto moderado**. "
                "El rendimiento de J28-J38 puede explicarse en parte por el modelo base "
                "y en parte por el contexto. Se recomienda EXP-6 para separar ambas contribuciones."
            )
        else:
            L.append(
                "El contexto tiene **impacto bajo** en este periodo. "
                "El rendimiento de J28-J38 parece atribuible principalmente al modelo base."
            )

    L.append("")
    L.append("### Próximos pasos")
    L.append(
        "- **EXP-6** (cuando Supabase esté activo): re-correr backtest con `arbitro_input` real → separa contribución de árbitro del overlay narrativo"
    )
    L.append(
        "- **EXP-7**: backtest con `--narrative-dir` para overlay retroactivo y medir impacto aislado"
    )

    return "\n".join(L)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Minando runs/*/prediction/prediction.json ...", file=sys.stderr)
    run_recs = mine_runs(RUNS_DIR)
    n_runs = len(set(r["run_id"] for r in run_recs))
    print(f"  {len(run_recs)} predicciones en {n_runs} runs", file=sys.stderr)

    print("Cargando backtest CSV ...", file=sys.stderr)
    bt_rows = load_backtest(BACKTEST_CSV)
    print(f"  {len(bt_rows)} filas", file=sys.stderr)

    print("Matcheando apuestas con runs ...", file=sys.stderr)
    matched = match_bets(bt_rows, run_recs)
    print(f"  {len(matched)} apuestas matcheadas", file=sys.stderr)

    if matched:
        fieldnames = list(matched[0].keys())
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            w.writeheader()
            w.writerows(matched)
        print(f"  CSV guardado: {OUTPUT_CSV}", file=sys.stderr)

    report = analyze(matched)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Reporte guardado: {OUTPUT_MD}", file=sys.stderr)

    print(report)


if __name__ == "__main__":
    main()
