"""
Extracts the fbref schedule links function for reuse across scripts.

Extracted verbatim from scripts/resultados_jornada.py.
"""

from __future__ import annotations

import importlib.util as _ilu
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_tm_spec = _ilu.spec_from_file_location("team_mapping", _HERE / "team_mapping.py")
_tm = _ilu.module_from_spec(_tm_spec)  # type: ignore[arg-type]
_tm_spec.loader.exec_module(_tm)  # type: ignore[union-attr]
normalize = _tm.normalize

FBREF_BASE = "https://fbref.com"
FBREF_SCHEDULE = f"{FBREF_BASE}/en/comps/12/schedule/La-Liga-Scores-and-Fixtures"


def wait_cloudflare(page, timeout: int = 30) -> bool:
    """Espera hasta que la página no sea el challenge de Cloudflare."""
    for _ in range(timeout):
        title = page.title().lower()
        if (
            "just a moment" not in title
            and "cloudflare" not in title
            and len(title) > 3
        ):
            return True
        time.sleep(1)
    return False


def get_schedule_links(page, season: str | None = None) -> dict[str, str]:
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
        print(
            "[WARN] Timeout esperando links de partidos — puede que la página no cargó"
        )

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
            _cal_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        else:
            _cal_date = raw_date[:10]

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
        for k, _v in sample:
            print(f"    {k}")
    return links
