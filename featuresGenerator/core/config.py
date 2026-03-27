"""Lee secretos de .env y parámetros de config.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# _BASE_DIR apunta a featuresGenerator/ (padre de core/)
_BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(_BASE_DIR / ".env")

# --- Secretos (.env) ---------------------------------------------------------
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]

# --- Parámetros del modelo (config.yaml) --------------------------------------
with open(_BASE_DIR / "config.yaml", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

SEASONS: list[int] = _cfg["seasons"]
DECAY_LAMBDA: float = _cfg["decay_lambda"]

PESO_FALTAS: float = _cfg["pesos"]["faltas"]
PESO_AMARILLAS: float = _cfg["pesos"]["amarillas"]
PESO_ROJAS: float = _cfg["pesos"]["rojas"]

ALPHA_CARD_PRESSURE: float = _cfg["alpha_card_pressure"]

# ── Knowledge Pack ─────────────────────────────────────────────────────────
FORMA_VENTANA: int = _cfg.get("forma_ventana", 5)
HOME_GOALS_FACTOR: float = _cfg.get("home_goals_factor", 1.07)
JORNADAS_LALIGA: int = _cfg.get("jornadas_laliga", 38)
MARKET_ALIGNMENT_THRESHOLD: float = _cfg.get("market_alignment_threshold", 0.60)

_umbrales_match = _cfg.get("umbrales_match", {})
FISICO_MUY_ALTO: float = _umbrales_match.get("fisico_muy_alto", 28.0)
FISICO_ALTO: float = _umbrales_match.get("fisico_alto", 24.0)
FISICO_NORMAL: float = _umbrales_match.get("fisico_normal", 20.0)
OFENSIVO_MUY_ABIERTO: float = _umbrales_match.get("ofensivo_muy_abierto", 26.0)
OFENSIVO_ABIERTO: float = _umbrales_match.get("ofensivo_abierto", 22.0)
OFENSIVO_EQUILIBRADO: float = _umbrales_match.get("ofensivo_equilibrado", 17.0)

_agg_vol = _cfg.get("agresividad_volumen", {})
AGG_VOL_PESO_FALTAS: float = _agg_vol.get("peso_faltas", 1.0)
AGG_VOL_PESO_AMARILLAS: float = _agg_vol.get("peso_amarillas", 0.10)
AGG_VOL_PESO_ROJAS: float = _agg_vol.get("peso_rojas", 0.00)
