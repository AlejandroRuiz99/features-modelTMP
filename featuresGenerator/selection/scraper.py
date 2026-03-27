"""
Scraper de posesión de La Liga desde fbref.com.

Usa undetected-chromedriver para pasar Cloudflare.
Proceso lento (~2-4h para una temporada completa) por rate limiting.

Escribe directamente a Supabase. En re-ejecuciones salta partidos
que ya tienen posesión, así que es seguro interrumpir y retomar.

Uso:
  python main.py --scrape
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import date, datetime

from selection.team_mapping import normalize

FBREF_BASE = "https://fbref.com"
_CURRENT_CALENDAR = f"{FBREF_BASE}/en/comps/12/schedule/La-Liga-Scores-and-Fixtures"


def _calendar_url(season_start: int) -> str:
    """
    Devuelve la URL del calendario de fbref para una temporada dada.
    season_start=2025 → temporada actual (URL sin año).
    season_start=2024 → 2024-2025-La-Liga-Scores-and-Fixtures.
    season_start=2023 → 2023-2024-La-Liga-Scores-and-Fixtures.
    """
    current_year = date.today().year
    if season_start >= current_year:
        return _CURRENT_CALENDAR
    slug = f"{season_start}-{season_start + 1}"
    return f"{FBREF_BASE}/en/comps/12/{slug}/schedule/{slug}-La-Liga-Scores-and-Fixtures"


def _create_driver():
    import undetected_chromedriver as uc

    opts = uc.ChromeOptions()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    return uc.Chrome(options=opts)


def _wait_for_page(driver, timeout: int = 20) -> bool:
    for _ in range(timeout):
        title = driver.title.lower()
        if "moment" not in title and "momento" not in title and len(driver.title) > 3:
            return True
        time.sleep(1)
    return False


def _extract_date(soup, url: str = "") -> str:
    """Extrae la fecha del partido (YYYY-MM-DD) de un match report de fbref."""
    el = soup.find(attrs={"data-venue-date": True})
    if el:
        return el["data-venue-date"]

    meta = soup.find("div", class_="scorebox_meta")
    if meta:
        for a in meta.find_all("a"):
            href = a.get("href", "")
            for part in href.split("/"):
                if len(part) == 10 and part[4:5] == "-" and part[7:8] == "-":
                    try:
                        datetime.strptime(part, "%Y-%m-%d")
                        return part
                    except ValueError:
                        pass

    if url:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", url)
        if m:
            return m.group(1)

    return ""


def _scrape_season(driver, season_start: int, already_done: set) -> tuple[int, int]:
    """
    Scrapea posesión de una temporada concreta.
    Devuelve (escritos, saltados).
    """
    from bs4 import BeautifulSoup
    from selection.supabase_client import update_possession_single

    url = _calendar_url(season_start)
    season_label = f"{season_start}-{season_start + 1}"
    print(f"\n--- Temporada {season_label} ---")
    print(f"    {url}")

    driver.get(url)
    if not _wait_for_page(driver):
        print("x No se pudo pasar Cloudflare en calendario")
        return 0, 0

    time.sleep(1.5)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Extraer (fecha, local, visitante, link) desde la tabla del calendario.
    # Usamos td[data-stat="match_report"] para encontrar los links (selector
    # fiable que ya funcionaba), y subimos al tr para extraer fecha y equipos.
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

    match_entries = []
    for td_report in soup.select('td[data-stat="match_report"]'):
        a = td_report.find("a")
        if not a or not a.get("href"):
            continue
        link = FBREF_BASE + a["href"]

        tr = td_report.parent  # fila del calendario
        td_date = tr.find(attrs={"data-stat": "date"})
        raw_date = (td_date.get("csk") or td_date.get_text(strip=True)) if td_date else ""
        cal_date = raw_date[:10] if len(raw_date) >= 10 else raw_date

        cal_home = normalize(_row_text(tr, *_HOME_STATS))
        cal_away = normalize(_row_text(tr, *_AWAY_STATS))
        match_entries.append((cal_date, cal_home, cal_away, link))

    total = len(match_entries)
    print(f"    {total} partidos encontrados\n")

    written = 0
    skipped = 0

    for i, (cal_date, cal_home, cal_away, link) in enumerate(match_entries):
        # Saltar ANTES de cargar la página si ya tenemos los datos en Supabase
        if cal_date and cal_home and (cal_date, cal_home, cal_away) in already_done:
            skipped += 1
            continue

        print(f"  [{i + 1:>3}/{total}] ", end="", flush=True)
        try:
            driver.get(link)
            if not _wait_for_page(driver):
                print("x Cloudflare bloqueado")
                continue

            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, "html.parser")

            match_date = _extract_date(soup, link) or cal_date
            scorebox = soup.find("div", class_="scorebox")
            teams = (
                [a.text.strip() for a in scorebox.select('strong > a[href*="/squads/"]')]
                if scorebox
                else []
            )

            home_name = normalize(teams[0]) if teams else cal_home
            away_name = normalize(teams[1]) if len(teams) > 1 else cal_away

            if (match_date, home_name, away_name) in already_done:
                print(f"skip  {home_name} vs {away_name}")
                skipped += 1
                continue

            team_stats = soup.find("div", id="team_stats")
            if not team_stats:
                print("x  Sin team_stats")
                continue

            rows_ts = team_stats.find_all("tr")
            found = False
            for j, row in enumerate(rows_ts):
                th = row.find("th")
                if th and "Possession" in th.get_text():
                    if j + 1 < len(rows_ts):
                        strongs = rows_ts[j + 1].find_all("strong")
                        if len(strongs) >= 2:
                            home_poss = int(strongs[0].text.strip().replace("%", ""))
                            away_poss = int(strongs[1].text.strip().replace("%", ""))

                            ok = update_possession_single(
                                match_date, home_name, away_name,
                                home_poss, away_poss,
                            )
                            if ok:
                                print(f"ok    {home_name} {home_poss}% vs {away_poss}% {away_name}")
                                written += 1
                                already_done.add((match_date, home_name, away_name))
                            else:
                                print(f"x  No en BD: {home_name} vs {away_name} ({match_date})")
                            found = True
                    break

            if not found:
                print("x  Sin posesion en pagina")

        except Exception as e:
            print(f"x  Error: {e}")

        time.sleep(1.5)

    return written, skipped


def _seasons_with_missing_possession() -> dict[str, int]:
    """
    Consulta Supabase y devuelve {season_label: n_sin_posesion}
    solo para las temporadas que tienen partidos sin posesión.
    """
    from selection.supabase_client import get_client
    sb = get_client()
    rows = sb.table("matches").select("season,possession_home").execute()
    missing: Counter = Counter()
    for r in rows.data:
        if r.get("possession_home") is None:
            missing[r["season"]] += 1
    return dict(missing)


def scrape_possession(seasons: list[int] | None = None) -> int:
    """
    Scrapea posesion de todos los partidos de La Liga desde fbref
    y escribe cada resultado directamente a Supabase.

    Solo procesa temporadas que tienen partidos sin posesión en Supabase.
    Salta partidos individuales que ya tienen posesion (sin cargar la página).
    Es seguro interrumpir y retomar.
    Devuelve el total de partidos nuevos escritos.
    """
    from selection.supabase_client import get_matches_with_possession
    from core.config import SEASONS

    if seasons is None:
        seasons = SEASONS

    # Pre-check: qué temporadas tienen partidos sin posesión
    print("Consultando qué temporadas necesitan posesion...")
    missing_by_season = _seasons_with_missing_possession()
    if not missing_by_season:
        print("   Todo completo, no hay nada que scrapear.")
        return 0
    for s, n in sorted(missing_by_season.items()):
        print(f"   {s}: {n} partidos sin posesion")

    # Filtrar solo las temporadas necesarias
    def _season_label(start: int) -> str:
        return f"{start}-{(start + 1) % 100:02d}"

    seasons_needed = [s for s in seasons if _season_label(s) in missing_by_season]
    if not seasons_needed:
        print("   Las temporadas con datos faltantes no están en la configuracion.")
        return 0

    print(f"\nTemporadas a procesar: {[_season_label(s) for s in seasons_needed]}")

    print("\nConsultando partidos con posesion ya en Supabase...")
    already_done = get_matches_with_possession()
    print(f"   {len(already_done)} partidos ya tienen posesion.\n")

    print("Iniciando Chrome...")
    driver = _create_driver()
    total_written = 0
    total_skipped = 0

    try:
        for season_start in seasons_needed:
            w, s = _scrape_season(driver, season_start, already_done)
            total_written  += w
            total_skipped  += s
            if season_start != seasons_needed[-1]:
                print("  Pausa entre temporadas...")
                time.sleep(4)

    finally:
        driver.quit()

    print(f"\nTotal: {total_written} partidos actualizados, {total_skipped} ya tenian posesion.")
    return total_written
