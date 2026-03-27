"""
Script completo: genera training data, entrena ensemble, predice faltas.
Uso: python run_prediction.py
"""
import sys
import json
from pathlib import Path

FEATURES_DIR = Path(__file__).parent / "featuresGenerator"
PRED_DIR = Path(__file__).parent / "predictionModels"
sys.path.insert(0, str(FEATURES_DIR))
sys.path.insert(0, str(PRED_DIR))

PARQUET_PATH = PRED_DIR / "data" / "training.parquet"
CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"
FEATURES_PATH = Path(__file__).parent / "features_rm_atm.json"


def step1_generate_parquet():
    print("\n" + "=" * 65)
    print("  PASO 1: Generando datos de entrenamiento desde Supabase")
    print("=" * 65)
    from training_data.generator import generate_from_supabase
    n = generate_from_supabase(PARQUET_PATH)
    print(f"  -> {n} partidos procesados")
    return n


def step2_train():
    print("\n" + "=" * 65)
    print("  PASO 2: Entrenando ensemble")
    print("=" * 65)
    from scripts.train import train
    ensemble = train(
        checkpoint_dir=CHECKPOINT_DIR,
        parquet_path=PARQUET_PATH,
        season_split="2025-26",
    )
    return ensemble


def step3_generate_features():
    print("\n" + "=" * 65)
    print("  PASO 3: Generando features para Real Madrid vs Atletico")
    print("         Arbitro: Munuera Montero | Jornada 29")
    print("=" * 65)
    from generate import generate_features

    feat = generate_features(
        equipo_local="Real Madrid",
        equipo_visitante="Ath Madrid",
        jornada=29,
        arbitro="Munuera Montero",
        fecha_partido="2026-03-22",
    )

    FEATURES_PATH.write_text(json.dumps(feat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Features guardadas en {FEATURES_PATH}")
    return feat


def step4_predict(feat, ensemble=None):
    print("\n" + "=" * 65)
    print("  PASO 4: Prediccion de faltas")
    print("=" * 65)
    from scripts.predict import _print_prediction
    from src.models.ensemble import FoulPredictionEnsemble

    if ensemble is None:
        cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text(encoding="utf-8"))
        ensemble = FoulPredictionEnsemble(config=cfg)
        ensemble.load(CHECKPOINT_DIR)

    _print_prediction(ensemble, feat)


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        step1_generate_parquet()
    else:
        print(f"[INFO] Parquet ya existe: {PARQUET_PATH}")

    has_weights = (CHECKPOINT_DIR / "regression.pt").exists()
    if not has_weights:
        ensemble = step2_train()
    else:
        print(f"[INFO] Ensemble ya entrenado en {CHECKPOINT_DIR}")
        ensemble = None

    feat = step3_generate_features()
    step4_predict(feat, ensemble)
