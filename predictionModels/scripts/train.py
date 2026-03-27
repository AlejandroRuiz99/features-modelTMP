"""
train.py — Pipeline de entrenamiento del ensemble desde Parquet.

Uso:
  python -m scripts.train --parquet data/training.parquet --checkpoint checkpoints/ensemble/
  python -m scripts.train --parquet data/training.parquet --season-split 2024-25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.models.ensemble import FoulPredictionEnsemble

# Columnas que son labels (target) y no deben incluirse en las features
_LABEL_COLS = {"fouls_total", "fouls_home", "fouls_away"}


# Valores por defecto para features ausentes (CSV o JSON incompleto)
FEATURE_DEFAULTS = {
    "home_team": "unknown",
    "away_team": "unknown",
    "referee": "unknown",
    "matchday": 19,
    "season": "2025-26",
    "date": "",
    "home_fouls_committed_avg": 12.0,
    "home_fouls_suffered_avg": 12.0,
    "away_fouls_committed_avg": 12.0,
    "away_fouls_suffered_avg": 12.0,
    "home_fouls_committed_curr": 12.0,
    "away_fouls_committed_curr": 12.0,
    "home_shots_curr": 11.0,
    "away_shots_curr": 11.0,
    "home_corners_curr": 4.5,
    "away_corners_curr": 4.5,
    "home_yellows_avg": 2.0,
    "away_yellows_avg": 2.0,
    "home_reds_avg": 0.1,
    "away_reds_avg": 0.1,
    "home_rank_hist": 10.0,
    "away_rank_hist": 10.0,
    "home_rank_curr": 10,
    "away_rank_curr": 10,
    "rank_diff_norm": 0.0,
    "season_phase": 0.5,
    "is_derby": False,
    "pace_index_curr": 31.0,
    "home_possession": 0.5,
    "away_possession": 0.5,
    "home_xg": 0.0,
    "away_xg": 0.0,
    "xg_diff": 0.0,
    "xfouls_home": 12.5,
    "xfouls_away": 12.5,
    "aggressiveness_volume_home": 0.5,
    "aggressiveness_volume_away": 0.5,
    "aggressiveness_norm_total": 0.5,
    "fouls_provoked_home": 12.0,
    "fouls_provoked_away": 12.0,
    "forma_fouls_home": 12.0,
    "forma_fouls_away": 12.0,
    "urgency_home": 0.5,
    "urgency_away": 0.5,
    "momentum_home": 0.5,
    "momentum_away": 0.5,
    "fatigue_home": 0.2,
    "fatigue_away": 0.2,
    "days_rest_home": 7.0,
    "days_rest_away": 7.0,
    "xfouls_factor_home": 1.0,
    "xfouls_factor_away": 1.0,
    "referee_mu_permisivo": 22.0,
    "referee_mu_estricto": 30.0,
    "referee_sigma_permisivo": 4.0,
    "referee_sigma_estricto": 4.0,
    "referee_peso_estricto": 0.5,
    "referee_n_partidos": 0,
    "ref_home_delta": 0.0,
    "ref_away_delta": 0.0,
    "ref_pair_delta_sum": 0.0,
    "ref_pair_samples": 0.0,
    "has_market_odds": False,
    "market_home_win_prob": 1/3,
    "market_draw_prob": 1/3,
    "market_away_win_prob": 1/3,
    "market_favorite_prob": 1/3,
    "market_balance": 1.0,
    "market_entropy": 1.0986,
    "market_ou25_over_prob": 0.50,
    "market_ou25_under_prob": 0.50,
    "intensidad_esperada": "media",
    "riesgo_disciplinario": "medio",
    "h2h_faltas_media": 25.0,
    "h2h_partidos": 0,
}


# ---------------------------------------------------------------------------
# Carga desde Parquet (formato óptimo — features ya aplanadas por training.pipeline)
# ---------------------------------------------------------------------------

def load_training_parquet(parquet_path: Path) -> list[dict]:
    """
    Carga feature dicts desde un archivo Parquet generado por el modulo training.

    El Parquet ya contiene las features aplanadas + las columnas de labels
    (fouls_total, fouls_home, fouls_away).

    Ventajas respecto a JSONs:
      - Carga 10-50x más rápida (lectura columnar, sin parseo de texto)
      - 3-5x menos uso de disco (compresión Snappy)
      - Tipos de datos correctos sin conversión manual
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas es requerido para cargar Parquet. pip install pandas pyarrow")

    df = pd.read_parquet(parquet_path, engine="pyarrow")

    required = {"fouls_total", "fouls_home", "fouls_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El Parquet no contiene las columnas de labels: {missing}")

    # Filtrar filas sin fouls_total válido
    df = df[df["fouls_total"].notna() & (df["fouls_total"] > 0)].copy()

    # rank_diff_norm: derivar si no existe
    if "rank_diff_norm" not in df.columns and "home_rank_curr" in df.columns:
        df["rank_diff_norm"] = (
            (df["home_rank_curr"] - df["away_rank_curr"]).abs() / 19.0
        )

    # Rellenar valores ausentes con defaults
    for col, default in FEATURE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    feature_dicts = df.to_dict(orient="records")
    print(f"[INFO] {len(feature_dicts)} partidos cargados de {parquet_path} "
          f"({len(df.columns)} columnas)")
    return feature_dicts


def split_train_gating(
    feature_dicts: list[dict],
    season_split: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Divide los feature dicts en set de entrenamiento y set de gating.

    Si season_split = "2024-25":
      - train: temporadas anteriores
      - gating: temporada especificada (ya jugada, out-of-sample)
    Si no, usa 80/20 cronologico.
    """
    if season_split:
        train = [f for f in feature_dicts if f.get("season") != season_split]
        gating = [f for f in feature_dicts if f.get("season") == season_split]
        print(f"[INFO] Train: {len(train)} partidos | Gating (out-of-sample): {len(gating)} partidos (temporada {season_split})")
    else:
        n = len(feature_dicts)
        split = int(n * 0.80)
        train = feature_dicts[:split]
        gating = feature_dicts[split:]
        print(f"[INFO] Train: {len(train)} partidos | Gating (out-of-sample): {len(gating)} partidos (split 80/20 cronologico)")

    if not train:
        raise ValueError("El set de entrenamiento esta vacio.")
    if not gating:
        print("[WARN] Gating set vacio. El gating se entrenara in-sample (riesgo de sobreajuste).")

    return train, gating


def build_team_stats(feature_dicts: list[dict]) -> tuple[dict, dict, dict]:
    """
    Calcula medias de equipo necesarias para NaiveBayes desde los feature dicts.

    Returns:
        avg_committed, avg_suffered, avg_rank
    """
    committed: dict[str, list] = {}
    suffered: dict[str, list] = {}
    rank: dict[str, list] = {}

    for f in feature_dicts:
        home = f.get("home_team", "")
        away = f.get("away_team", "")

        committed.setdefault(home, []).append(f.get("home_fouls_committed_avg", 12.0))
        committed.setdefault(away, []).append(f.get("away_fouls_committed_avg", 12.0))
        suffered.setdefault(home, []).append(f.get("home_fouls_suffered_avg", 12.0))
        suffered.setdefault(away, []).append(f.get("away_fouls_suffered_avg", 12.0))
        rank.setdefault(home, []).append(f.get("home_rank_curr", 10))
        rank.setdefault(away, []).append(f.get("away_rank_curr", 10))

    avg_c = {t: sum(v) / len(v) for t, v in committed.items()}
    avg_s = {t: sum(v) / len(v) for t, v in suffered.items()}
    avg_r = {t: sum(v) / len(v) for t, v in rank.items()}

    return avg_c, avg_s, avg_r


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(
    checkpoint_dir: Optional[Path] = None,
    parquet_path: Optional[Path] = None,
    feature_dicts: Optional[list[dict]] = None,
    season_split: Optional[str] = None,
    config: Optional[dict] = None,
) -> FoulPredictionEnsemble:
    """
    Entrena el ensemble. Fuente de datos:
    - feature_dicts: lista directa (ej. desde API)
    - parquet_path:  Parquet con features aplanadas (RECOMENDADO)
    """
    if feature_dicts is None and parquet_path is None:
        raise ValueError("Indica --parquet o feature_dicts")
    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints/ensemble/")

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO DEL ENSEMBLE")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # 1. Cargar feature dicts
    if feature_dicts is not None:
        pass  # ya proporcionados
    elif parquet_path is not None:
        feature_dicts = load_training_parquet(parquet_path)
    else:
        raise ValueError("Solo se admite fuente Parquet para entrenamiento.")

    if len(feature_dicts) < 20:
        print(f"[WARN] Solo {len(feature_dicts)} partidos disponibles.")
    if not feature_dicts:
        raise ValueError("Ningun dato pudo convertirse a feature dict.")

    # 2. Dividir en train y gating
    train_feats, gating_feats = split_train_gating(feature_dicts, season_split)

    # 3. Estadisticas de equipo para NaiveBayes
    avg_committed, avg_suffered, avg_rank = build_team_stats(train_feats)

    # 4. Crear y entrenar el ensemble
    cfg = config or {
        "naive_bayes": {"n_clusters": 5, "percentiles": [25, 50, 75]},
        "regression": {
            "regularization": 0.01,
            "intercept_init": 3.2,
            "referee_offset_reference": 26.0,
            "use_extended_features": True,
            "use_ref_team_features": True,
            "use_context_features": True,
        },
        "anfis": {
            "n_membership_functions": 3,
            "lr": 0.005,
            "epochs": 150,
            "batch_size": 64,
            "use_market_context": True,
            "output_center": 25.0,
            "output_scale": 18.0,
            "output_tanh_temp": 4.0,
            "match_intensity_weights": [0.25, 0.20, 0.25, 0.15, 0.15],
            "play_style_weights": [0.40, 0.35, 0.25],
        },
        "gating_network": {
            "hidden_dims": [48, 24],
            "dropout": 0.15,
            "temperature": 1.0,
            "lr": 0.001,
            "min_weight": 0.03,
            "prior_mix": 0.45,
        },
        "reconciliation": {"total_variance_scale": 0.6},
    }

    ensemble = FoulPredictionEnsemble(config=cfg)

    print("[INFO] Entrenando ensemble...")
    ensemble.fit(
        matches=train_feats,
        team_avg_committed=avg_committed,
        team_avg_suffered=avg_suffered,
        team_avg_rank=avg_rank,
        fit_team_models=True,
        gating_matches=gating_feats or None,
    )

    # 5. Guardar checkpoint y config (para que predict cargue la misma arquitectura)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    ensemble.save(checkpoint_dir)
    print(f"\n[OK] Ensemble guardado en {checkpoint_dir}")

    # 6. Quick eval in-sample
    errors = []
    for f in train_feats[:min(50, len(train_feats))]:
        try:
            pred = ensemble.predict(f)
            errors.append(abs(pred.expected_fouls - f["fouls_total"]))
        except Exception:
            continue

    if errors:
        mae = sum(errors) / len(errors)
        print(f"[INFO] MAE in-sample (primeros {len(errors)} partidos): {mae:.2f} faltas")

    print(f"\n{'='*60}")
    print(f"  Entrenamiento completado.")
    print(f"  Para predecir: python -m scripts.predict --input <json> --checkpoint {checkpoint_dir}")
    print(f"{'='*60}\n")

    return ensemble


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    checkpoint_dir = Path(args.checkpoint)
    parquet_path = Path(args.parquet)    if args.parquet    else None

    if parquet_path and not parquet_path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {parquet_path}")

    train(
        checkpoint_dir=checkpoint_dir,
        parquet_path=parquet_path,
        season_split=args.season_split,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento del ensemble de prediccion de faltas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python -m scripts.train --parquet data/training.parquet\n"
            "  python -m scripts.train --parquet data/training.parquet --season-split 2024-25\n"
        ),
    )
    parser.add_argument(
        "--parquet", type=str, default=None,
        help=(
            "Archivo Parquet con features aplanadas (generado por python -m training). "
            "RECOMENDADO: carga 10-50x más rápida que JSONs individuales."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/ensemble/",
        help="Directorio donde guardar el ensemble entrenado. Default: checkpoints/ensemble/",
    )
    parser.add_argument(
        "--season-split", type=str, default=None,
        help=(
            "Temporada out-of-sample para el gating (ej: 2024-25). "
            "Si no se especifica, se usa split 80/20 cronologico."
        ),
    )
    args = parser.parse_args()

    # Default: buscar parquet antes que JSONs
    if not args.parquet:
        default_parquet = Path("data/training.parquet")
        if default_parquet.exists():
            args.parquet = str(default_parquet)
            print(f"[INFO] Usando Parquet por defecto: {default_parquet}")
        else:
            raise FileNotFoundError("No se encontró data/training.parquet")

    main(args)
