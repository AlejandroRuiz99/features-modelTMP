"""
backtest_odds.py - Evalua el ensemble contra las cuotas de odds_definitivo_25_26.csv.

Walk-forward correcto: para cada partido, el state se reconstruye usando solo
partidos con fecha estrictamente anterior al kickoff. Sin fuga temporal.

Mercados evaluados (3 por partido):
  - Total Fouls           -> pmf total (reconciliado si hay team_pred)
  - Team Total Fouls (L)  -> home_pmf
  - Team Total Fouls (V)  -> away_pmf

Criterio de apuesta:
  - compute_ev(line, p_over, odds_over, odds_under, min_edge)
  - Si retorna EVResult != None -> apuesta 1 unidad al lado indicado (over|under)

PnL por apuesta (lineas .5, sin push):
  - actual > line, bet=over  -> ganada: +odds-1
  - actual < line, bet=under -> ganada: +odds-1
  - en otro caso             -> perdida: -1

Uso:
  python scripts/backtest_odds.py --source CODERE --min-edge 0.05
  python scripts/backtest_odds.py --source ALL --min-edge 0.03 --out reports/backtest_all.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# sys.path bootstrap (igual que run_prediction.py)
ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features_generator"
PRED_DIR = ROOT / "prediction_models"
sys.path.insert(0, str(PRED_DIR))
sys.path.insert(0, str(FEATURES_DIR))

from assembly import build_features  # noqa: E402
from core.state_cache import build_state  # noqa: E402
from src.models.ensemble import FoulPredictionEnsemble  # noqa: E402
from src.utils.ev import compute_ev  # noqa: E402

CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"
CSV_PATH = ROOT / "odds_definitivo_25_26.csv"

logger = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def _to_float_opt(s: str | None) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _parse_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parsea el CSV (semicolon, decimal-comma).

    Devuelve (rows, stats) donde stats trackea filas descartadas por datos
    incompletos (cuotas faltantes).
    """
    rows: list[dict[str, Any]] = []
    stats = {"completed": 0, "skipped_missing_odds": 0, "skipped_missing_line": 0}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r.get("estado") != "completed":
                continue
            stats["completed"] += 1
            linea = _to_float_opt(r.get("linea"))
            odds_over = _to_float_opt(r.get("cuota_over"))
            odds_under = _to_float_opt(r.get("cuota_under"))
            if linea is None:
                stats["skipped_missing_line"] += 1
                continue
            if odds_over is None or odds_under is None:
                stats["skipped_missing_odds"] += 1
                continue
            rows.append(
                {
                    "fixture_id": r["fixture_id"],
                    "jornada": int(r["jornada"]),
                    "kickoff": r["kickoff"],
                    "date": r["kickoff"][:10],
                    "home": r["equipo_local"],
                    "away": r["equipo_visitante"],
                    "mercado": r["mercado"],
                    "equipo_lado": (r.get("equipo") or "").strip(),
                    "linea": linea,
                    "odds_over": odds_over,
                    "odds_under": odds_under,
                    "fuente": r["fuente"],
                }
            )
    return rows, stats


def _group_by_fixture(rows: list[dict]) -> list[dict[str, Any]]:
    """Agrupa filas por fixture_id. Devuelve lista ordenada por kickoff."""
    by_fix: dict[str, dict[str, Any]] = {}
    for r in rows:
        f = by_fix.setdefault(
            r["fixture_id"],
            {
                "fixture_id": r["fixture_id"],
                "jornada": r["jornada"],
                "kickoff": r["kickoff"],
                "date": r["date"],
                "home": r["home"],
                "away": r["away"],
                "fuente": r["fuente"],
                "markets": [],
            },
        )
        f["markets"].append(
            {
                "mercado": r["mercado"],
                "equipo_lado": r["equipo_lado"],
                "linea": r["linea"],
                "odds_over": r["odds_over"],
                "odds_under": r["odds_under"],
            }
        )
    return sorted(by_fix.values(), key=lambda x: x["kickoff"])


def _build_actuals_index(
    all_partidos: list[dict],
) -> tuple[dict[tuple, tuple[int, int]], set[str], dict[tuple, str]]:
    """Devuelve (idx, canonical_teams, referee_idx).

    idx: {(date10, canonical_home, canonical_away) -> (fouls_home, fouls_away)}.
    canonical_teams: nombres canonicos cortos (los de Supabase).
    referee_idx: {(date10, canonical_home, canonical_away) -> referee_name}.
    """
    idx: dict[tuple, tuple[int, int]] = {}
    ref_idx: dict[tuple, str] = {}
    teams: set[str] = set()
    for p in all_partidos:
        d = (p.get("date") or "")[:10]
        h = (p.get("home") or {}).get("name") or ""
        a = (p.get("away") or {}).get("name") or ""
        if h:
            teams.add(h)
        if a:
            teams.add(a)
        fh = int((p.get("home") or {}).get("fouls") or 0)
        fa = int((p.get("away") or {}).get("fouls") or 0)
        idx[(d, h, a)] = (fh, fa)
        ref = (p.get("referee") or "").strip()
        if ref:
            ref_idx[(d, h, a)] = ref
    return idx, teams, ref_idx


def _resolve_team(name: str, canonical: list[str], cache: dict[str, str]) -> str | None:
    """Resuelve un nombre del CSV al canonico via fuzzy_name_search. Cachea."""
    if name in cache:
        return cache[name]
    from core.utils import TEAM_ALIASES, fuzzy_name_search

    resolved = fuzzy_name_search(name, canonical, TEAM_ALIASES)
    cache[name] = resolved or ""
    return resolved or None


# ---------------------------------------------------------------------------
# Walk-forward state
# ---------------------------------------------------------------------------


def _state_for_date(
    all_partidos: list[dict], cutoff_date: str, calendar_rows: list[dict] | None
) -> dict:
    """Construye state usando solo partidos con date < cutoff_date."""
    pre = [p for p in all_partidos if (p.get("date") or "")[:10] < cutoff_date]
    return build_state(pre, calendar_rows=calendar_rows)


# ---------------------------------------------------------------------------
# Evaluacion por mercado
# ---------------------------------------------------------------------------


def _p_over_from_pmf(pmf: Any, line: float) -> float | None:
    """Saca p_over para una linea desde un PMF (usa over_under_table)."""
    if pmf is None:
        return None
    try:
        table = pmf.over_under_table([line])
        if line in table:
            return float(table[line][0])
    except Exception:
        return None
    return None


def _eval_market(
    pmf: Any,
    line: float,
    odds_over: float,
    odds_under: float,
    actual: int,
    min_edge: float,
) -> dict | None:
    """Evalua un mercado. Devuelve dict con la apuesta (si la hay) y el PnL."""
    p_over = _p_over_from_pmf(pmf, line)
    if p_over is None:
        return None
    ev = compute_ev(
        line=line,
        p_over_model=p_over,
        odds_over=odds_over,
        odds_under=odds_under,
        kelly_fraction=0.25,
        min_edge=min_edge,
    )
    if ev is None:
        return None

    bet_over = ev["bet"] == "over"
    won = (actual > line) if bet_over else (actual < line)
    pnl = (ev["odds"] - 1.0) if won else -1.0
    return {
        "p_over": p_over,
        "bet": ev["bet"],
        "edge": ev["edge"],
        "odds": ev["odds"],
        "actual": actual,
        "won": won,
        "pnl": pnl,
    }


# ---------------------------------------------------------------------------
# Backtest principal
# ---------------------------------------------------------------------------


def run_backtest(
    fixtures: list[dict],
    all_partidos: list[dict],
    calendar_rows: list[dict] | None,
    ensemble: FoulPredictionEnsemble,
    actuals_idx: dict,
    canonical_teams: list[str],
    min_edge: float,
    referee_idx: dict | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Ejecuta el backtest walk-forward. Devuelve (bets, stats)."""
    bets: list[dict] = []
    stats = {
        "processed": 0,
        "skipped_no_result": 0,
        "skipped_name_resolve": 0,
        "skipped_pipeline_error": 0,
    }
    name_cache: dict[str, str] = {}
    n = len(fixtures)
    t0 = time.time()

    for i, fx in enumerate(fixtures, 1):
        elapsed = time.time() - t0
        eta = (elapsed / i) * (n - i) if i > 0 else 0
        logger.info(
            "[%d/%d] %s %s vs %s (J%d)  | elapsed %ds eta %ds",
            i,
            n,
            fx["date"],
            fx["home"],
            fx["away"],
            fx["jornada"],
            int(elapsed),
            int(eta),
        )

        home_can = _resolve_team(fx["home"], canonical_teams, name_cache)
        away_can = _resolve_team(fx["away"], canonical_teams, name_cache)
        if not home_can or not away_can:
            logger.warning(
                "  Nombres no resolubles: %s | %s -> skip", fx["home"], fx["away"]
            )
            stats["skipped_name_resolve"] += 1
            continue

        actuals = actuals_idx.get((fx["date"], home_can, away_can))
        if actuals is None:
            logger.warning(
                "  Sin resultado real (%s, %s, %s) -> skip",
                fx["date"],
                home_can,
                away_can,
            )
            stats["skipped_no_result"] += 1
            continue
        actual_home, actual_away = actuals
        actual_total = actual_home + actual_away

        # Walk-forward state
        try:
            state = _state_for_date(all_partidos, fx["date"], calendar_rows)
        except Exception as exc:
            logger.error("  build_state fallo: %s", exc)
            stats["skipped_pipeline_error"] += 1
            continue

        # Features + prediccion
        referee_name = None
        arb_source = "not_available"
        if referee_idx is not None:
            referee_name = referee_idx.get((fx["date"], home_can, away_can))
            if referee_name:
                arb_source = "historical"
        try:
            feat = build_features(
                state=state,
                equipo_local_input=home_can,
                equipo_visitante_input=away_can,
                jornada=fx["jornada"],
                arbitro_input=referee_name,
                arbitraje_source=arb_source,
                fecha_partido_input=fx["date"],
                skip_market_fetch=True,
            )
        except Exception as exc:
            logger.error("  build_features fallo: %s", exc)
            stats["skipped_pipeline_error"] += 1
            continue

        try:
            pred = ensemble.predict(feat)
            team_pred = (
                ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True)
                or {}
            )
        except Exception as exc:
            logger.error("  predict fallo: %s", exc)
            stats["skipped_pipeline_error"] += 1
            continue
        stats["processed"] += 1

        total_pmf = team_pred.get("total_pmf") or getattr(pred, "pmf_total", None)
        home_pmf = team_pred.get("home_pmf")
        away_pmf = team_pred.get("away_pmf")

        # Por cada mercado del fixture, evaluar
        for m in fx["markets"]:
            mercado = m["mercado"]
            if mercado == "Total Fouls":
                pmf = total_pmf
                actual = actual_total
                side_label = "total"
            elif mercado == "Team Total Fouls" and m["equipo_lado"] == "local":
                pmf = home_pmf
                actual = actual_home
                side_label = "local"
            elif mercado == "Team Total Fouls" and m["equipo_lado"] == "visitante":
                pmf = away_pmf
                actual = actual_away
                side_label = "visitante"
            else:
                continue

            res = _eval_market(
                pmf=pmf,
                line=m["linea"],
                odds_over=m["odds_over"],
                odds_under=m["odds_under"],
                actual=actual,
                min_edge=min_edge,
            )
            if res is None:
                continue

            bets.append(
                {
                    "fixture_id": fx["fixture_id"],
                    "date": fx["date"],
                    "jornada": fx["jornada"],
                    "home": fx["home"],
                    "away": fx["away"],
                    "fuente": fx["fuente"],
                    "referee": referee_name or "",
                    "mercado": mercado,
                    "side_label": side_label,
                    "linea": m["linea"],
                    "p_over_model": round(res["p_over"], 4),
                    "edge": round(res["edge"], 4),
                    "bet": res["bet"],
                    "odds": res["odds"],
                    "actual": res["actual"],
                    "won": int(res["won"]),
                    "pnl": round(res["pnl"], 4),
                }
            )

    return bets, stats


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


def _summarize(bets: list[dict]) -> dict:
    if not bets:
        return {"n": 0, "wins": 0, "losses": 0, "hit_rate": 0.0, "pnl": 0.0, "roi": 0.0}
    wins = sum(b["won"] for b in bets)
    losses = len(bets) - wins
    pnl = sum(b["pnl"] for b in bets)
    return {
        "n": len(bets),
        "wins": wins,
        "losses": losses,
        "hit_rate": wins / len(bets),
        "pnl": pnl,
        "roi": pnl / len(bets),  # stake = 1 ud/apuesta
    }


def _print_report(bets: list[dict], min_edge: float) -> None:
    print("\n" + "=" * 78)
    print(f"BACKTEST RESULTS (min_edge = {min_edge:.2%})")
    print("=" * 78)

    overall = _summarize(bets)
    print(
        f"\nOVERALL: n={overall['n']} | wins={overall['wins']} | "
        f"losses={overall['losses']} | hit-rate={overall['hit_rate']:.2%} | "
        f"PnL={overall['pnl']:+.2f}u | ROI={overall['roi']:+.2%}"
    )

    # Por fuente
    by_src: dict[str, list] = defaultdict(list)
    for b in bets:
        by_src[b["fuente"]].append(b)
    print("\nPOR FUENTE:")
    for src, group in sorted(by_src.items()):
        s = _summarize(group)
        print(
            f"  {src:30s} n={s['n']:4d} | hit={s['hit_rate']:6.2%} | "
            f"PnL={s['pnl']:+7.2f}u | ROI={s['roi']:+7.2%}"
        )

    # Por mercado x lado
    by_mkt: dict[str, list] = defaultdict(list)
    for b in bets:
        key = (
            b["mercado"]
            if b["mercado"] == "Total Fouls"
            else f"{b['mercado']} ({b['side_label']})"
        )
        by_mkt[key].append(b)
    print("\nPOR MERCADO:")
    for mkt, group in sorted(by_mkt.items()):
        s = _summarize(group)
        print(
            f"  {mkt:35s} n={s['n']:4d} | hit={s['hit_rate']:6.2%} | "
            f"PnL={s['pnl']:+7.2f}u | ROI={s['roi']:+7.2%}"
        )

    # Por lado (over/under)
    by_side: dict[str, list] = defaultdict(list)
    for b in bets:
        by_side[b["bet"]].append(b)
    print("\nPOR LADO:")
    for side, group in sorted(by_side.items()):
        s = _summarize(group)
        print(
            f"  {side:10s} n={s['n']:4d} | hit={s['hit_rate']:6.2%} | "
            f"PnL={s['pnl']:+7.2f}u | ROI={s['roi']:+7.2%}"
        )

    print("=" * 78)


def _write_csv(bets: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "fixture_id",
        "date",
        "jornada",
        "home",
        "away",
        "fuente",
        "referee",
        "mercado",
        "side_label",
        "linea",
        "p_over_model",
        "edge",
        "bet",
        "odds",
        "actual",
        "won",
        "pnl",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for b in bets:
            w.writerow(b)
    logger.info("Bets guardadas en %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_ensemble() -> FoulPredictionEnsemble:
    import json

    cfg = (CHECKPOINT_DIR / "config.json").read_text(encoding="utf-8")
    ensemble = FoulPredictionEnsemble(config=json.loads(cfg))
    ensemble.load(CHECKPOINT_DIR)
    return ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=["CODERE", "CONSENSUS_CODERE_STYLE", "ALL"],
        default="CODERE",
        help="Filtrar por fuente de cuotas. Default: CODERE (real).",
    )
    ap.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Edge minimo para apostar (default: 0.05 = production).",
    )
    ap.add_argument(
        "--out",
        default="reports/backtest_bets.csv",
        help="Ruta CSV de salida con detalle de apuestas.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limita N fixtures (debug).")
    ap.add_argument(
        "--with-referee",
        action="store_true",
        default=False,
        help="EXP-6: inyecta el árbitro real (de Supabase) en build_features.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # CSV
    logger.info("Parseando %s...", CSV_PATH)
    rows, parse_stats = _parse_csv(CSV_PATH)
    logger.info(
        "CSV: completed=%d, descartadas (sin cuotas)=%d, descartadas (sin linea)=%d, validas=%d",
        parse_stats["completed"],
        parse_stats["skipped_missing_odds"],
        parse_stats["skipped_missing_line"],
        len(rows),
    )
    if args.source != "ALL":
        rows = [r for r in rows if r["fuente"] == args.source]
    fixtures = _group_by_fixture(rows)
    if args.limit > 0:
        fixtures = fixtures[: args.limit]
    logger.info(
        "Fixtures fuente=%s: %d | mercados: %d",
        args.source,
        len(fixtures),
        len(rows),
    )

    # Estado historico
    logger.info("Cargando partidos + calendario desde Supabase...")
    from selection import supabase_client

    all_partidos = supabase_client.fetch_all_matches()
    try:
        calendar_rows = supabase_client.fetch_liga_calendar()
    except Exception:
        calendar_rows = None
    logger.info("Partidos historicos: %d", len(all_partidos))

    actuals_idx, canonical_set, referee_idx_raw = _build_actuals_index(all_partidos)
    canonical_teams = sorted(canonical_set)
    referee_idx = referee_idx_raw if args.with_referee else None
    if args.with_referee:
        n_with_ref = len(referee_idx_raw)
        logger.info(
            "EXP-6 mode: referee injection ON (%d partidos con árbitro)", n_with_ref
        )
    logger.info(
        "Resultados reales indexados: %d | equipos canonicos: %d",
        len(actuals_idx),
        len(canonical_teams),
    )

    # Ensemble
    logger.info("Cargando ensemble...")
    ensemble = _load_ensemble()

    # Run
    bets, run_stats = run_backtest(
        fixtures=fixtures,
        all_partidos=all_partidos,
        calendar_rows=calendar_rows,
        ensemble=ensemble,
        actuals_idx=actuals_idx,
        canonical_teams=canonical_teams,
        min_edge=args.min_edge,
        referee_idx=referee_idx,
    )

    logger.info(
        "Fixtures procesados: %d | sin resultado: %d | name-resolve fallo: %d | pipeline fallo: %d",
        run_stats["processed"],
        run_stats["skipped_no_result"],
        run_stats["skipped_name_resolve"],
        run_stats["skipped_pipeline_error"],
    )

    # Output
    _write_csv(bets, ROOT / args.out)
    _print_report(bets, args.min_edge)


if __name__ == "__main__":
    main()
