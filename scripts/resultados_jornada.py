"""
Compara predicciones de faltas con resultados reales de la jornada.

Carga partidos + árbitros de un JSON de jornada, predicciones del modelo,
y usa Playwright para scrapeear las faltas reales de fbref.com.
Imprime una tabla comparativa.

Uso:
    python scripts/resultados_jornada.py
    python scripts/resultados_jornada.py --partidos partidos_jornada30.json --predicciones jornada30_predicciones.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Importar normalize directamente desde el módulo (sin pasar por __init__.py que carga supabase)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "team_mapping",
    ROOT / "features_generator" / "selection" / "team_mapping.py",
)
_tm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
normalize = _tm.normalize

FBREF_BASE = "https://fbref.com"
FBREF_SCHEDULE = f"{FBREF_BASE}/en/comps/12/schedule/La-Liga-Scores-and-Fixtures"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_partidos(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_predicciones(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_match_key(home: str, away: str) -> str:
    return f"{normalize(home).lower()}|{normalize(away).lower()}"


def merge_data(partidos: list[dict], predicciones: list[dict]) -> list[dict]:
    """
    Une partidos (árbitro + fecha completa) con predicciones (faltas modelo).
    Key de unión: par (local_canonical, visitante_canonical).
    """
    pred_index: dict[str, dict] = {}
    for p in predicciones:
        home_p, away_p = p["match"].split(" vs ", 1)
        key = f"{home_p.strip().lower()}|{away_p.strip().lower()}"
        pred_index[key] = p

    rows = []
    for partido in partidos:
        home_can = normalize(partido["local"])
        away_can = normalize(partido["visitante"])
        key = f"{home_can.lower()}|{away_can.lower()}"
        pred = pred_index.get(key, {})
        rows.append({
            "local": home_can,
            "visitante": away_can,
            "fecha": partido["fecha"],
            "arbitro": partido.get("arbitro", ""),
            "pred_local": pred.get("home_expected"),
            "pred_visitante": pred.get("away_expected"),
            "pred_total": pred.get("expected_fouls"),
        })
    return rows


# ---------------------------------------------------------------------------
# Playwright scraping
# ---------------------------------------------------------------------------

def wait_cloudflare(page, timeout: int = 30) -> bool:
    """Espera hasta que la página no sea el challenge de Cloudflare."""
    for _ in range(timeout):
        title = page.title().lower()
        if "just a moment" not in title and "cloudflare" not in title and len(title) > 3:
            return True
        time.sleep(1)
    return False


def get_schedule_links(page) -> dict[str, str]:
    """
    Carga la página del calendario de fbref y devuelve
    {canonical_key: match_report_url}  donde key = "home_can|away_can"
    """
    print("Cargando calendario fbref...")
    page.goto(FBREF_SCHEDULE, timeout=60_000)
    if not wait_cloudflare(page):
        raise RuntimeError("Cloudflare bloqueó el calendario")

    # Esperar a que aparezca al menos un enlace de match report (renderizado por JS)
    try:
        page.wait_for_selector('td[data-stat="match_report"] a', timeout=20_000)
    except Exception:
        print("[WARN] Timeout esperando links de partidos — puede que la página no cargó")

    time.sleep(1)

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page.content(), "html.parser")

    links: dict[str, str] = {}
    _HOME_STATS = ("home_team", "squad_a", "home_squad")
    _AWAY_STATS = ("away_team", "squad_b", "away_squad")

    def _row_text(tr, *stats):
        for s in stats:
            el = tr.find(attrs={"data-stat": s})
            if el:
                t = el.get_text(strip=True)
                if t:
                    return t
        return ""

    for td_report in soup.select('td[data-stat="match_report"]'):
        a = td_report.find("a")
        if not a or not a.get("href"):
            continue
        tr = td_report.parent

        td_date = tr.find(attrs={"data-stat": "date"})
        if not td_date:
            continue
        raw_date = (td_date.get("csk") or td_date.get_text(strip=True)).strip()
        if raw_date and "-" not in raw_date and len(raw_date) >= 8:
            cal_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        else:
            cal_date = raw_date[:10]

        home_raw = _row_text(tr, *_HOME_STATS)
        away_raw = _row_text(tr, *_AWAY_STATS)
        home_can = normalize(home_raw).lower()
        away_can = normalize(away_raw).lower()
        if home_can and away_can:
            key = f"{home_can}|{away_can}"
            links[key] = FBREF_BASE + a["href"]

    print(f"  {len(links)} partidos encontrados en el calendario")
    if links:
        # Mostrar muestra para debug
        sample = list(links.items())[:3]
        for k, v in sample:
            print(f"    {k}")
    return links


def scrape_fouls(page, url: str, match_label: str) -> tuple[int | None, int | None]:
    """Entra al match report y extrae (fouls_home, fouls_away)."""
    print(f"  Scrapeando {match_label}...")
    page.goto(url, timeout=60_000)
    if not wait_cloudflare(page):
        print(f"  [WARN] Cloudflare bloqueó {match_label}")
        return None, None
    time.sleep(1.5)

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page.content(), "html.parser")

    # Estrategia 1: div#team_stats con fila "Fouls"
    team_stats = soup.find("div", id="team_stats")
    if team_stats:
        rows_ts = team_stats.find_all("tr")
        for i, row in enumerate(rows_ts):
            th = row.find("th")
            if th and "fouls" in th.get_text(strip=True).lower():
                if i + 1 < len(rows_ts):
                    next_row = rows_ts[i + 1]
                    strongs = next_row.find_all("strong")
                    if len(strongs) >= 2:
                        try:
                            return int(strongs[0].text.strip()), int(strongs[1].text.strip())
                        except ValueError:
                            pass

    # Estrategia 2: buscar en tablas de stats individuales la columna "fouls"
    # (stats_h_summary / stats_a_summary)
    home_fouls = _extract_fouls_from_summary(soup, "home")
    away_fouls = _extract_fouls_from_summary(soup, "away")
    if home_fouls is not None and away_fouls is not None:
        return home_fouls, away_fouls

    print(f"  [WARN] No se encontraron faltas para {match_label}")
    return None, None


def _extract_fouls_from_summary(soup, side: str) -> int | None:
    """
    Busca la suma de faltas cometidas en la tabla stats_{side}_summary de fbref.
    """
    table_id = f"stats_{side}_summary"
    table = soup.find("table", {"id": table_id})
    if not table:
        return None

    total_row = table.find("tfoot")
    if not total_row:
        return None

    fouls_td = total_row.find("td", attrs={"data-stat": "fouls"})
    if fouls_td:
        txt = fouls_td.get_text(strip=True)
        try:
            return int(txt)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    header = (
        f"{'Partido':<30} {'Árbitro':<28} "
        f"{'P.Local':>7} {'P.Visita':>8} {'P.Total':>7}  "
        f"{'R.Local':>7} {'R.Visita':>8} {'R.Total':>7}  "
        f"{'Error':>6}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        match_label = f"{r['local']} vs {r['visitante']}"
        arbitro = r["arbitro"][:27] if r["arbitro"] else "-"
        pl = f"{r['pred_local']:.1f}" if r["pred_local"] is not None else "-"
        pv = f"{r['pred_visitante']:.1f}" if r["pred_visitante"] is not None else "-"
        pt = f"{r['pred_total']:.1f}" if r["pred_total"] is not None else "-"

        rl = f"{r['real_local']}" if r["real_local"] is not None else "-"
        rv = f"{r['real_visitante']}" if r["real_visitante"] is not None else "-"
        rt_val = r["real_total"]
        rt = f"{rt_val}" if rt_val is not None else "-"

        if r["pred_total"] is not None and rt_val is not None:
            error = r["pred_total"] - rt_val
            err_str = f"{error:+.1f}"
        else:
            err_str = "-"

        print(
            f"{match_label:<30} {arbitro:<28} "
            f"{pl:>7} {pv:>8} {pt:>7}  "
            f"{rl:>7} {rv:>8} {rt:>7}  "
            f"{err_str:>6}"
        )
    print(sep)

    # Resumen
    errors = [
        r["pred_total"] - r["real_total"]
        for r in rows
        if r["pred_total"] is not None and r["real_total"] is not None
    ]
    if errors:
        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        print(f"\nPartidos con resultado: {len(errors)}/{len(rows)}")
        print(f"MAE  (error absoluto medio): {mae:.2f} faltas")
        print(f"Bias (error medio con signo): {bias:+.2f} faltas")


def save_results(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compara predicciones vs resultados reales de faltas")
    parser.add_argument("--partidos", default="partidos_jornada30.json")
    parser.add_argument("--predicciones", default="jornada30_predicciones.json")
    parser.add_argument("--output", default="resultados_jornada30.json")
    parser.add_argument("--headless", action="store_true", help="Lanzar Chrome en modo headless (puede fallar con Cloudflare)")
    return parser.parse_args()


def main():
    args = parse_args()

    partidos = load_partidos(args.partidos)
    predicciones = load_predicciones(args.predicciones)
    rows = merge_data(partidos, predicciones)

    print(f"\nJornada {partidos[0]['jornada']} — {len(rows)} partidos")
    print("Pred: predicción modelo | Real: resultado fbref\n")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=args.headless,
            channel="chrome",   # usa Chrome instalado (mejor contra Cloudflare)
            args=["--window-size=1280,900", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/134.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        # Aplicar stealth para evitar detección de bot
        try:
            from playwright_stealth import stealth_sync
            stealth_sync(page)
            print("Stealth mode activo")
        except ImportError:
            pass

        # Cargar todos los links del calendario de una vez
        schedule_links = get_schedule_links(page)

        # Scrapeear faltas para cada partido
        for row in rows:
            key = f"{row['local'].lower()}|{row['visitante'].lower()}"
            url = schedule_links.get(key)
            if not url:
                print(f"  [SKIP] {row['local']} vs {row['visitante']} — no encontrado en calendario")
                row["real_local"] = None
                row["real_visitante"] = None
                row["real_total"] = None
                continue

            label = f"{row['local']} vs {row['visitante']}"
            home_f, away_f = scrape_fouls(page, url, label)
            row["real_local"] = home_f
            row["real_visitante"] = away_f
            row["real_total"] = (home_f + away_f) if (home_f is not None and away_f is not None) else None

            time.sleep(1.5)  # rate limiting

        browser.close()

    print_table(rows)
    save_results(rows, args.output)


if __name__ == "__main__":
    main()
