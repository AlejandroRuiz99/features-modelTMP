"""Generador del dataset de entrenamiento (Parquet).

Pipeline:
    Supabase (partidos + objectives + calendario)
    → estado historico (scores, arbitros, GMM)
    → features por partido
    → Parquet

Uso:
    python -m training_data
    python -m training_data --output ../predictionModels/data/training.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assembly import build_features
from assembly.betting_odds import market_features_from_historical_odds
from core.state_cache import build_state
from selection import supabase_client

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
        f_h  = p.get("home", {}).get("fouls")
        f_a  = p.get("away", {}).get("fouls")
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

def _build_row(match: MatchRecord, state: dict) -> dict:
    feat = build_features(
        state=state,
        equipo_local_input=match.home,
        equipo_visitante_input=match.away,
        arbitro_input=match.referee,
        jornada=match.jornada,
        fecha_partido_input=match.fecha,
        skip_market_fetch=True,
        features_profile="training",
    )
    feat.update(market_features_from_historical_odds(match.raw_odds))
    feat["fouls_total"] = float(match.fouls_total)
    feat["fouls_home"]  = float(match.fouls_home)
    feat["fouls_away"]  = float(match.fouls_away)
    feat["matchday"]    = match.jornada
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
    print(f"[OK] {output_path}  ({len(df)} filas x {len(df.columns)} cols, {size_kb:.1f} KB)")


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
    print("[1/4] Cargando partidos desde Supabase...")
    partidos = supabase_client.fetch_all_matches()
    if not partidos:
        print("[ERROR] No hay partidos en Supabase")
        return 0
    print(f"      {len(partidos)} partidos")

    print("[2/4] Cargando objectives y calendario...")
    try:
        objectives = supabase_client.fetch_laliga_objectives()
    except Exception as e:
        print(f"      [WARN] Sin objectives: {e}")
        objectives = {}
    try:
        cal_rows = supabase_client.fetch_liga_calendar()
        print(f"      {len(cal_rows)} filas de calendario")
    except Exception as e:
        print(f"      [WARN] Sin calendario: {e}")
        cal_rows = None

    # -- 2. Estado historico ------------------------------------------------
    print("[3/4] Construyendo estado historico...")
    state = build_state(partidos, objectives=objectives, calendar_rows=cal_rows)
    print(
        f"      {len(state.get('ref_perfiles', {}))} arbitros | "
        f"{len(state.get('perfiles_gmm', {}))} GMM | "
        f"{len(state.get('cal_index', {}))} equipos en calendario"
    )

    # -- 3. Features --------------------------------------------------------
    print(f"[4/4] Generando features ({len(partidos)} partidos)...")
    rows: list[dict] = []
    skipped = errors = 0

    for i, p in enumerate(partidos, 1):
        match = MatchRecord.from_raw(p)
        if match is None or not match.is_sane():
            skipped += 1
            continue
        try:
            rows.append(_build_row(match, state))
        except Exception as e:
            errors += 1
            print(f"      [ERR] {match.home} vs {match.away}: {e}")
        if i % 200 == 0:
            print(f"      {i}/{len(partidos)} — OK:{len(rows)} skip:{skipped} err:{errors}")

    # -- 4. Guardar ---------------------------------------------------------
    if rows:
        _save_parquet(rows, output_path)

    print(f"\nResumen: {len(rows)} filas | {skipped} sin datos | {errors} errores")
    return len(rows)
