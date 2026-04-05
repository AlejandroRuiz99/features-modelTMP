"""Generador del dataset de entrenamiento (Parquet).

Pipeline:
    Supabase (partidos + objectives + calendario)
    → estado historico (scores, arbitros, GMM)
    → features por partido
    → Parquet

Uso:
    python -m training_data
    python -m training_data --output ../prediction_models/data/training.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from assembly import build_features

logger = logging.getLogger(__name__)
from assembly.betting_odds import market_features_from_historical_odds
from core.state_cache import build_state
from selection import supabase_client


# ---------------------------------------------------------------------------
# Helpers de matchday y season_phase
# ---------------------------------------------------------------------------


def _season_phase_from_date(fecha: str, season: int) -> float:
    """Fracción 0-1 de la temporada transcurrida en la fecha del partido.

    La Liga arranca ~1 de agosto y termina ~30 de junio del año siguiente.
    """
    from datetime import date as _date

    try:
        d = _date.fromisoformat(fecha[:10])
        start = _date(season, 8, 1)
        end = _date(season + 1, 6, 30)
        total_days = (end - start).days
        elapsed = (d - start).days
        return round(max(0.0, min(1.0, elapsed / total_days)), 4)
    except Exception:
        return 0.5


def _build_team_match_counts(
    accumulated: list[dict],
) -> dict[tuple[str, str], int]:
    """Devuelve {(team, season_str): n_partidos_jugados} sobre el acumulado hasta hoy."""
    counts: dict[tuple[str, str], int] = {}
    for p in accumulated:
        season_str = str(p.get("season", ""))
        for side in ("home", "away"):
            team = p.get(side, {}).get("name", "")
            if team:
                key = (team, season_str)
                counts[key] = counts.get(key, 0) + 1
    return counts

_FOULS_SANITY_MIN = 10


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRecord:
    """Partido historico validado y tipado."""

    home: str
    away: str
    fouls_home: int
    fouls_away: int
    fecha: str
    referee: str | None
    season: int | None
    jornada: int | None
    raw_odds: dict

    @property
    def fouls_total(self) -> int:
        return self.fouls_home + self.fouls_away

    @property
    def season_label(self) -> str | None:
        if self.season is None:
            return None
        return f"{self.season}-{(self.season + 1) % 100:02d}"

    def is_sane(self) -> bool:
        return self.fouls_total >= _FOULS_SANITY_MIN

    @classmethod
    def from_raw(cls, p: dict) -> MatchRecord | None:
        """Parsea dict crudo de Supabase. Retorna None si faltan campos criticos."""
        home = p.get("home", {}).get("name")
        away = p.get("away", {}).get("name")
        f_h = p.get("home", {}).get("fouls")
        f_a = p.get("away", {}).get("fouls")
        if not home or not away or f_h is None or f_a is None:
            return None
        raw_jornada = p.get("jornada")
        return cls(
            home=home,
            away=away,
            fouls_home=int(f_h or 0),
            fouls_away=int(f_a or 0),
            fecha=str(p.get("date", ""))[:10],
            referee=p.get("referee"),
            season=p.get("season"),
            jornada=int(raw_jornada) if raw_jornada is not None else None,
            raw_odds=p.get("odds") or {},
        )


# ---------------------------------------------------------------------------
# Privados
# ---------------------------------------------------------------------------


def _build_row(
    match: MatchRecord,
    state: dict,
    team_match_counts: dict[tuple[str, str], int] | None = None,
) -> dict:
    # Estimar jornada si no viene de Supabase: nº de partidos previos del local + 1
    jornada = match.jornada
    if jornada is None and team_match_counts is not None and match.season is not None:
        season_str = str(match.season)
        jornada = team_match_counts.get((match.home, season_str), 0) + 1

    feat = build_features(
        state=state,
        equipo_local_input=match.home,
        equipo_visitante_input=match.away,
        arbitro_input=match.referee,
        jornada=jornada,
        fecha_partido_input=match.fecha,
        skip_market_fetch=True,
        features_profile="training",
    )
    feat.update(market_features_from_historical_odds(match.raw_odds))
    feat["fouls_total"] = float(match.fouls_total)
    feat["fouls_home"] = float(match.fouls_home)
    feat["fouls_away"] = float(match.fouls_away)
    feat["matchday"] = jornada

    # season_phase desde fecha real — más fiable que jornada / 38
    if match.season is not None:
        feat["season_phase"] = _season_phase_from_date(match.fecha, match.season)

    if match.season_label:
        feat["season"] = match.season_label
    return feat


def _save_parquet(rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if "is_derby" in df.columns:
        df["is_derby"] = df["is_derby"].astype(bool)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy", engine="pyarrow")
    size_kb = output_path.stat().st_size / 1024
    logger.info(
        "[OK] %s  (%d filas x %d cols, %.1f KB)",
        output_path,
        len(df),
        len(df.columns),
        size_kb,
    )


# ---------------------------------------------------------------------------
# Entrada publica
# ---------------------------------------------------------------------------


def run(output_path: Path) -> int:
    """Ejecuta el pipeline completo de training desde Supabase.

    1. Descarga partidos, objectives y calendario en una sola sesion.
    2. Construye el estado historico (scores, arbitros, GMM, calendario).
    3. Genera features para cada partido valido.
    4. Guarda el Parquet resultante.

    Returns:
        Numero de filas guardadas en el Parquet.
    """
    # -- 1. Ingesta ---------------------------------------------------------
    logger.info("[1/4] Cargando partidos desde Supabase...")
    partidos = supabase_client.fetch_all_matches()
    if not partidos:
        logger.error("No hay partidos en Supabase")
        return 0
    logger.info("      %d partidos", len(partidos))

    logger.info("[2/4] Cargando objectives y calendario...")
    try:
        objectives = supabase_client.fetch_laliga_objectives()
    except Exception as e:
        logger.warning("Sin objectives: %s", e, exc_info=True)
        objectives = {}
    try:
        cal_rows = supabase_client.fetch_liga_calendar()
        logger.info("      %d filas de calendario", len(cal_rows))
    except Exception as e:
        logger.warning("Sin calendario: %s", e, exc_info=True)
        cal_rows = None

    # -- 2. Estado historico ------------------------------------------------
    # (el estado se construye incrementalmente en el paso 3, walk-forward por fecha)
    logger.info("[3/4] Estado historico: se construira walk-forward por fecha.")

    # -- 3. Features (walk-forward por fecha para evitar data leakage) --------
    # Cada partido solo puede ver partidos anteriores a su propia fecha.
    # Se construye el estado una vez por fecha unica (~500 builds vs 1500).
    logger.info("[4/4] Generando features (%d partidos, walk-forward)...", len(partidos))
    rows: list[dict] = []
    skipped = errors = 0

    partidos_sorted = sorted(
        partidos,
        key=lambda p: (p.get("date") or "")[:10],
    )

    # Agrupar por fecha (YYYY-MM-DD)
    from collections import defaultdict as _defaultdict

    by_date: dict[str, list[dict]] = _defaultdict(list)
    for p in partidos_sorted:
        by_date[(p.get("date") or "")[:10]].append(p)

    accumulated: list[dict] = []
    total = len(partidos)
    processed = 0

    for date in sorted(by_date):
        batch = by_date[date]

        if accumulated:
            # Estado construido SOLO con partidos anteriores a esta fecha
            state = build_state(
                accumulated, objectives=objectives, calendar_rows=cal_rows
            )
        # Si accumulated esta vacio (primeros partidos del dataset) no hay
        # estado historico suficiente — se saltean.

        # Conteo de partidos previos por equipo para estimar jornada
        team_counts = _build_team_match_counts(accumulated) if accumulated else {}

        for p in batch:
            processed += 1
            if not accumulated:
                skipped += 1
                continue
            match = MatchRecord.from_raw(p)
            if match is None or not match.is_sane():
                skipped += 1
                continue
            try:
                rows.append(_build_row(match, state, team_match_counts=team_counts))
            except Exception as e:
                errors += 1
                logger.warning(
                    "[ERR] %s vs %s: %s", match.home, match.away, e, exc_info=True
                )

        accumulated.extend(batch)

        if processed % 200 == 0 or processed == total:
            logger.info(
                "      %d/%d — OK:%d skip:%d err:%d",
                processed,
                total,
                len(rows),
                skipped,
                errors,
            )

    # -- 4. Guardar ---------------------------------------------------------
    if rows:
        _save_parquet(rows, output_path)

    logger.info(
        "Resumen: %d filas | %d sin datos | %d errores", len(rows), skipped, errors
    )
    return len(rows)
