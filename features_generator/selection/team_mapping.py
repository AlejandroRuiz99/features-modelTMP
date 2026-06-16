"""
Normalización de nombres de equipo entre fuentes de datos.

Nombre canónico = el que usa football-data.co.uk (nuestro estándar en Supabase).
"""

from __future__ import annotations

FBREF_TO_DB: dict[str, str] = {
    # --- Nombres oficiales largos ---
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Atlético de Madrid": "Ath Madrid",
    "Atletico de Madrid": "Ath Madrid",
    "Rayo Vallecano": "Vallecano",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "RCD Espanyol": "Espanol",
    "Espanyol": "Espanol",
    "Deportivo Alavés": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Alavés": "Alaves",
    "Cádiz": "Cadiz",
    "Cadiz CF": "Cadiz",
    "UD Las Palmas": "Las Palmas",
    "RCD Mallorca": "Mallorca",
    "Celta Vigo": "Celta",
    "Celta de Vigo": "Celta",
    "RC Celta": "Celta",
    "Real Valladolid": "Valladolid",
    "CD Leganés": "Leganes",
    "CD Leganes": "Leganes",
    "Leganés": "Leganes",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "CA Osasuna": "Osasuna",
    "FC Barcelona": "Barcelona",
    "Granada CF": "Granada",
    "UD Almería": "Almeria",
    "Almería": "Almeria",
    "Real Oviedo": "Oviedo",
    "Elche CF": "Elche",
    "Levante UD": "Levante",
    "Racing de Santander": "Racing Santander",
}


# Short-form aliases used by Codere (scraped by odds_raw) that FBREF_TO_DB
# does not cover. Codere shortens team names in its home_team/away_team
# columns, e.g. "Athletic" instead of "Athletic Club", "Atlético" instead of
# "Atlético de Madrid", "Rayo" instead of "Rayo Vallecano". Without these
# aliases, name matching against odds_raw rows fails silently.
#
# Keys are compared case-insensitively after stripping whitespace. Values
# are the canonical DB names (same vocabulary as FBREF_TO_DB values).
BOOKMAKER_SHORT_ALIASES: dict[str, str] = {
    "athletic": "Ath Bilbao",
    "atletico": "Ath Madrid",
    "atlético": "Ath Madrid",
    "rayo": "Vallecano",
}


def normalize(name: str) -> str:
    """Normaliza un nombre de equipo al formato canónico de la BD.

    Aplica dos mapeos en orden:
      1. Short-form aliases usados por bookmakers (p.ej. Codere → "Athletic"
         en vez de "Athletic Club") — comparación case-insensitive.
      2. Mapeo principal FBREF_TO_DB — comparación case-sensitive.

    Los nombres que no coinciden con ningún alias se devuelven tal cual
    (tras strip), lo cual preserva el comportamiento existente para
    nombres ya canónicos (p.ej. "Barcelona", "Valencia").
    """
    stripped = name.strip()
    # Short-form aliases first (case-insensitive)
    short = BOOKMAKER_SHORT_ALIASES.get(stripped.lower())
    if short is not None:
        return short
    # Main long-form mapping (case-sensitive, preserves existing behavior)
    return FBREF_TO_DB.get(stripped, stripped)
