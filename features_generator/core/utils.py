from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
import logging
from typing import Any


# Mapeo de alias de entrada (usuario/scrapers) → nombre canónico corto usado en la BD.
# Fuente única: iap.buscar_equipo y referees.buscar_arbitro usan fuzzy_name_search con este dict.
TEAM_ALIASES: dict[str, str] = {
    "athletic": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao",
    "bilbao": "Ath Bilbao",
    "atletico": "Ath Madrid",
    "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid",
    "atleti": "Ath Madrid",
    "barca": "Barcelona",
    "barça": "Barcelona",
    "fcb": "Barcelona",
    "madrid": "Real Madrid",
    "rmadrid": "Real Madrid",
    "rayo": "Vallecano",
    "rayo vallecano": "Vallecano",
    "real sociedad": "Sociedad",
    "la real": "Sociedad",
    "real betis": "Betis",
    "espanyol": "Espanol",
    "rcd espanyol": "Espanol",
    "deportivo alaves": "Alaves",
    "alavés": "Alaves",
    "cadiz": "Cadiz",
    "cádiz": "Cadiz",
    "ud las palmas": "Las Palmas",
    "rcd mallorca": "Mallorca",
    "rc celta": "Celta",
    "celta vigo": "Celta",
    "pucela": "Valladolid",
    "real valladolid": "Valladolid",
    "leganés": "Leganes",
    "cd leganes": "Leganes",
}


logger = logging.getLogger(__name__)


def fuzzy_name_search(
    query: str,
    candidates: list[str],
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Busqueda de nombre con fallback progresivo.

    Orden: alias exacto → igualdad normalizada → substring (query en candidato)
           → substring (candidato en query) → alias fuzzy.
    """
    q = norm_text(query)
    if not q:
        return None

    if aliases:
        direct = aliases.get(q)
        if direct and direct in candidates:
            return direct

    norm_candidates = {norm_text(c): c for c in candidates}

    if q in norm_candidates:
        return norm_candidates[q]

    matches = [c for nc, c in norm_candidates.items() if q in nc]
    if matches:
        return min(matches, key=len)

    matches = [c for nc, c in norm_candidates.items() if nc in q]
    if matches:
        return max(matches, key=len)

    if aliases:
        for alias_key, alias_val in aliases.items():
            nk = norm_text(alias_key)
            if q in nk or nk in q:
                if alias_val in candidates:
                    return alias_val

    return None


def no_vig(cuota_a: float, cuota_b: float) -> tuple[float, float]:
    """Elimina el margen y devuelve probabilidades reales para dos resultados."""
    if cuota_a <= 1.01 or cuota_b <= 1.01:
        return 0.5, 0.5
    inv_a, inv_b = 1.0 / cuota_a, 1.0 / cuota_b
    total = inv_a + inv_b
    return round(inv_a / total, 4), round(inv_b / total, 4)


def no_vig_3(c1: float, cx: float, c2: float) -> tuple[float, float, float]:
    """Elimina el margen para tres resultados (1X2)."""
    if any(c <= 1.01 for c in (c1, cx, c2)):
        return 0.40, 0.25, 0.35
    inv = [1.0 / c for c in (c1, cx, c2)]
    total = sum(inv)
    return round(inv[0] / total, 4), round(inv[1] / total, 4), round(inv[2] / total, 4)


def market_entropy(p_h: float, p_d: float, p_a: float) -> float:
    """Entropia de Shannon de la distribucion 1X2."""
    probs = [max(p, 1e-12) for p in (p_h, p_d, p_a)]
    return round(-sum(p * math.log(p) for p in probs), 4)


def norm_text(s: str | None) -> str:
    if not s:
        return ""
    out = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    out = out.lower().strip()
    return re.sub(r"\s+", " ", out)


def team_match(a: str | None, b: str | None) -> bool:
    na = norm_text(a)
    nb = norm_text(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    return SequenceMatcher(a=na, b=nb).ratio() >= 0.82


def extract_ou(selection: str | None) -> tuple[str | None, float | None]:
    txt = norm_text(selection)
    if not txt:
        return None, None
    side: str | None = None
    if "menos" in txt or "under" in txt:
        side = "under"
    elif "mas" in txt or "over" in txt:
        side = "over"

    m = re.search(r"(\d+(?:[.,]\d+)?)", txt)
    if not m:
        return side, None
    return side, float(m.group(1).replace(",", "."))


def safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f > 1.0 else None
    except Exception:
        logger.warning("safe_float: could not convert %r to float", v, exc_info=True)
        return None


def parse_date_safe(s: str) -> date | None:
    """Parsea YYYY-MM-DD a date, o None si falla."""
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        logger.warning("parse_date_safe: could not parse date %r", s, exc_info=True)
        return None


def classify_competition(competition: str | None) -> str:
    """Normaliza nombre de competicion a etiqueta corta."""
    c = (competition or "").lower()
    if not c:
        return "desconocida"
    if "champions" in c or "ucl" in c:
        return "ucl"
    if "conference" in c or "uecl" in c:
        return "uecl"
    if "europa" in c or "uel" in c:
        return "uel"
    if "copa" in c or "rey" in c or "cup" in c:
        return "copa"
    if "liga" in c or "laliga" in c or "la liga" in c:
        return "liga"
    return "otra"


def dedupe_dicts(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        sig = tuple(it.get(k) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(it)
    return out
