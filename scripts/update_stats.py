#!/usr/bin/env python3
"""
update_stats.py — Actualiza las estadísticas de La Liga a día de hoy.

Pipeline:
  1. Descarga los CSVs actualizados de football-data.co.uk para cada temporada
     configurada en features_generator/config.yaml.
  2. Sube los partidos nuevos a Supabase (idempotente: salta los ya existentes).
  3. Opcionalmente, rellena la posesión faltante vía scraper de fbref
     (lento: ~2-4h por temporada con Cloudflare).

Uso:
  python scripts/update_stats.py
  python scripts/update_stats.py --seasons 2024 2025
  python scripts/update_stats.py --scrape-possession
  python scripts/update_stats.py --only-possession
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path: habilita imports sin prefijo de paquete
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "features_generator"))
sys.path.insert(0, str(_ROOT / "prediction_models"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sync_csv(seasons: list[int]) -> None:
    """Descarga CSVs de football-data.co.uk y sube partidos nuevos a Supabase."""
    from selection.csv_source import download_season
    from selection.supabase_client import sync_new_matches

    total_insertados = 0
    total_omitidos = 0

    for season in seasons:
        logger.info("=" * 60)
        logger.info("Temporada %d/%d", season, season + 1)
        logger.info("=" * 60)

        records = download_season(season)
        if not records:
            logger.warning("  Sin datos para temporada %d.", season)
            continue

        insertados, omitidos = sync_new_matches(records)
        logger.info(
            "  Resultado: %d insertados, %d ya existían.",
            insertados,
            omitidos,
        )
        total_insertados += insertados
        total_omitidos += omitidos

    logger.info("-" * 60)
    logger.info(
        "TOTAL CSV: %d partidos insertados, %d ya existían.",
        total_insertados,
        total_omitidos,
    )


def _sync_possession(seasons: list[int]) -> None:
    """Rellena posesión faltante en Supabase vía scraper de fbref."""
    from selection.scraper import scrape_possession

    logger.info("=" * 60)
    logger.info("Scraper de posesión (fbref)")
    logger.info("Proceso lento (~2-4h). Se puede interrumpir y retomar.")
    logger.info("=" * 60)

    written = scrape_possession(seasons=seasons)
    logger.info("Posesión: %d partidos actualizados.", written)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Actualiza estadísticas de La Liga a día de hoy."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        metavar="YEAR",
        default=None,
        help="Temporadas a procesar (ej: 2024 2025). Por defecto: las de config.yaml.",
    )
    parser.add_argument(
        "--scrape-possession",
        action="store_true",
        help="Después de sincronizar CSVs, rellena posesión faltante vía fbref.",
    )
    parser.add_argument(
        "--only-possession",
        action="store_true",
        help="Salta el CSV y solo ejecuta el scraper de posesión.",
    )
    args = parser.parse_args()

    # Resolver temporadas
    if args.seasons:
        seasons = args.seasons
    else:
        from core.config import SEASONS
        seasons = SEASONS

    logger.info("Temporadas a procesar: %s", seasons)

    if not args.only_possession:
        _sync_csv(seasons)

    if args.scrape_possession or args.only_possession:
        _sync_possession(seasons)

    logger.info("Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
