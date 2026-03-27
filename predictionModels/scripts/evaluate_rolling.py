"""
evaluate_rolling.py — Evaluación del ensemble con ventana deslizante por jornadas.

Estrategias de evaluación implementadas:
  A) Season split:  entrenar en 2022-23 + 2023-24  ->  predecir 2024-25 completa
  B) Rolling intra-season: entrenar en años previos + primeras N jornadas de la temporada
                            target -> predecir las jornadas siguientes M a M.
                            Repite deslizando la ventana hasta final de temporada.

Las cuotas 1X2 y O/U 2.5 goles (B365/Pinnacle) SI están disponibles en datos
históricos y se inyectan en el Parquet desde la tabla matches.
Las señales de mercado de faltas (foul_market_*) NO están disponibles en datos
históricos; permanecen en 0.5/25.0 (sin señal) durante el entrenamiento y
solo se enriquecen con odds reales en tiempo de predicción desde odds_raw.

Uso:
  python -m scripts.evaluate_rolling --parquet data/training.parquet
  python -m scripts.evaluate_rolling --parquet data/training.parquet --strategy season
  python -m scripts.evaluate_rolling --parquet data/training.parquet --strategy rolling \\
         --target-season 2024-25 --train-jornadas 10 --eval-jornadas 10

Salida:
  MAE por ventana, MAE global, calibración Over/Under de las principales líneas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import (
    load_training_parquet,
    build_team_stats,
)
from src.models.ensemble import FoulPredictionEnsemble


# ---------------------------------------------------------------------------
# Métricas locales (sin sklearn)
# ---------------------------------------------------------------------------

def _mae(preds: list[float], reals: list[float]) -> float:
    if not preds:
        return float("nan")
    return sum(abs(p - r) for p, r in zip(preds, reals)) / len(preds)


def _rmse(preds: list[float], reals: list[float]) -> float:
    import math
    if not preds:
        return float("nan")
    return math.sqrt(sum((p - r) ** 2 for p, r in zip(preds, reals)) / len(preds))


def _bias(preds: list[float], reals: list[float]) -> float:
    if not preds:
        return float("nan")
    return sum(p - r for p, r in zip(preds, reals)) / len(preds)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load(parquet: Optional[Path]) -> list[dict]:
    if not parquet:
        raise ValueError("Se requiere --parquet")
    return load_training_parquet(parquet)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación de una ventana
# ---------------------------------------------------------------------------

def _train_and_eval(
    train_feats: list[dict],
    eval_feats: list[dict],
    checkpoint_dir: Optional[Path],
    config: dict,
    window_label: str,
) -> dict:
    if len(train_feats) < 20:
        print(f"  [SKIP] {window_label}: solo {len(train_feats)} partidos de entrenamiento")
        return {}

    avg_c, avg_s, avg_r = build_team_stats(train_feats)
    ensemble = FoulPredictionEnsemble(config=config)
    ensemble.fit(
        matches=train_feats,
        team_avg_committed=avg_c,
        team_avg_suffered=avg_s,
        team_avg_rank=avg_r,
        fit_team_models=True,
        gating_matches=None,
    )

    if checkpoint_dir:
        ckpt = checkpoint_dir / window_label.replace(" ", "_").replace("/", "-")
        ckpt.mkdir(parents=True, exist_ok=True)
        ensemble.save(ckpt)

    preds, reals = [], []
    for f in eval_feats:
        try:
            p = ensemble.predict(f)
            preds.append(p.expected_fouls)
            reals.append(f["fouls_total"])
        except Exception:
            continue

    return {
        "window":        window_label,
        "n_train":       len(train_feats),
        "n_eval":        len(eval_feats),
        "n_predichos":   len(preds),
        "mae":           round(_mae(preds, reals), 3),
        "rmse":          round(_rmse(preds, reals), 3),
        "bias":          round(_bias(preds, reals), 3),
    }


# ---------------------------------------------------------------------------
# Estrategia A: Season split (2022-23+2023-24 -> 2024-25)
# ---------------------------------------------------------------------------

def _season_key(s: str) -> int:
    """Convierte '2024-25' -> 2024, '2023-24' -> 2023. Para ordenar temporadas."""
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return 0


def evaluate_season_split(
    feature_dicts: list[dict],
    target_season: str,
    checkpoint_dir: Optional[Path],
    config: dict,
) -> list[dict]:
    target_key = _season_key(target_season)
    # Solo temporadas ANTERIORES al target (sin data leakage)
    train_feats = [f for f in feature_dicts if _season_key(f.get("season", "")) < target_key]
    eval_feats  = [f for f in feature_dicts if f.get("season") == target_season]

    print(f"\n[Season Split]  target={target_season}")
    seasons_train = sorted({f.get("season") for f in train_feats})
    print(f"  Train seasons: {seasons_train}  ({len(train_feats)} partidos)")
    print(f"  Eval:          {target_season}  ({len(eval_feats)} partidos)")

    if not eval_feats:
        print(f"  [WARN] No hay partidos de la temporada {target_season} en los datos.")
        return []

    result = _train_and_eval(
        train_feats, eval_feats, checkpoint_dir, config,
        window_label=f"season_split_{target_season}",
    )
    _print_window(result)
    return [result]


# ---------------------------------------------------------------------------
# Estrategia B: Rolling intra-season
# ---------------------------------------------------------------------------

def evaluate_rolling(
    feature_dicts: list[dict],
    target_season: str,
    train_jornadas: int,
    eval_jornadas: int,
    checkpoint_dir: Optional[Path],
    config: dict,
) -> list[dict]:
    """
    Desliza una ventana dentro de target_season:
      - Entrenamiento: temporadas anteriores + jornadas 1..N de target_season
      - Evaluación:    jornadas N+1..N+M de target_season
      - Desliza en pasos de eval_jornadas hasta el final de la temporada.
    """
    target_key = _season_key(target_season)
    previas = [f for f in feature_dicts if _season_key(f.get("season", "")) < target_key]
    target  = sorted(
        [f for f in feature_dicts if f.get("season") == target_season],
        key=lambda f: (f.get("matchday", 0), f.get("date", "")),
    )

    if not target:
        print(f"  [WARN] No hay partidos de la temporada {target_season} en los datos.")
        return []

    max_jornada = max(f.get("matchday", 0) for f in target)
    print(f"\n[Rolling {target_season}]  jornadas hasta {max_jornada}")
    print(f"  Ventana entreno: {train_jornadas} jornadas  |  ventana eval: {eval_jornadas} jornadas")
    print(f"  Partidos previos al target: {len(previas)}")

    results = []
    jornada_inicio_eval = train_jornadas + 1

    while jornada_inicio_eval <= max_jornada:
        jornada_fin_eval = min(jornada_inicio_eval + eval_jornadas - 1, max_jornada)
        jornada_fin_train = jornada_inicio_eval - 1

        train_target = [f for f in target if f.get("matchday", 0) <= jornada_fin_train]
        eval_window  = [f for f in target if jornada_inicio_eval <= f.get("matchday", 0) <= jornada_fin_eval]
        train_feats  = previas + train_target

        label = f"j{jornada_inicio_eval:02d}-{jornada_fin_eval:02d}_{target_season}"
        print(f"\n  Ventana {label}: train={len(train_feats)} | eval={len(eval_window)}")

        if len(eval_window) < 5:
            print(f"    [SKIP] Muy pocos partidos de evaluación ({len(eval_window)})")
            jornada_inicio_eval += eval_jornadas
            continue

        result = _train_and_eval(
            train_feats, eval_window, checkpoint_dir, config,
            window_label=label,
        )
        if result:
            results.append(result)
            _print_window(result)

        jornada_inicio_eval += eval_jornadas

    return results


# ---------------------------------------------------------------------------
# Impresión de resultados
# ---------------------------------------------------------------------------

def _print_window(r: dict) -> None:
    if not r:
        return
    print(f"    MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  bias={r['bias']:+.3f}  "
          f"(n_eval={r['n_predichos']})")


def _print_summary(results: list[dict]) -> None:
    if not results:
        print("\n[WARN] Sin resultados para mostrar.")
        return

    maes = [r["mae"] for r in results if r.get("n_predichos", 0) > 0]
    biases = [r["bias"] for r in results if r.get("n_predichos", 0) > 0]

    print("\n" + "=" * 58)
    print("  RESUMEN EVALUACION ROLLING")
    print("=" * 58)
    print(f"  Ventanas evaluadas:    {len(results)}")
    if maes:
        print(f"  MAE medio:             {sum(maes)/len(maes):.3f}")
        print(f"  MAE mejor ventana:     {min(maes):.3f}")
        print(f"  MAE peor ventana:      {max(maes):.3f}")
        print(f"  Bias medio:            {sum(biases)/len(biases):+.3f}")
    print(f"\n  Referencia MAE: <3.5=excelente | 3.5-5=bueno | >6=mejorable")
    print("=" * 58 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluación del ensemble con ventana deslizante",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Entrenar 2022-23+2023-24 -> evaluar 2024-25 completa\n"
            "  python -m scripts.evaluate_rolling --parquet data/training.parquet --strategy season --target-season 2024-25\n\n"
            "  # Primeras 10 jornadas de 2024-25 -> predecir las siguientes 10\n"
            "  python -m scripts.evaluate_rolling --parquet data/training.parquet --strategy rolling "
            "--target-season 2024-25 --train-jornadas 10 --eval-jornadas 10\n"
        ),
    )
    parser.add_argument("--parquet",         type=str, default=None)
    parser.add_argument(
        "--strategy",
        choices=["season", "rolling", "both"],
        default="both",
        help="Estrategia de evaluación (default: both)",
    )
    parser.add_argument(
        "--target-season", type=str, default="2024-25",
        help="Temporada de evaluación out-of-sample (default: 2024-25)",
    )
    parser.add_argument(
        "--train-jornadas", type=int, default=10,
        help="Jornadas del target_season usadas como entrenamiento inicial en rolling (default: 10)",
    )
    parser.add_argument(
        "--eval-jornadas", type=int, default=10,
        help="Tamaño de cada ventana de evaluación en rolling (default: 10)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Si se indica, guarda un checkpoint por ventana en este directorio.",
    )
    args = parser.parse_args(argv)

    # Fuente de datos
    if not args.parquet:
        default_parquet = Path("data/training.parquet")
        if default_parquet.exists():
            args.parquet = str(default_parquet)
        else:
            raise FileNotFoundError("No se encontró data/training.parquet")

    parquet_p  = Path(args.parquet)   if args.parquet  else None
    ckpt_p     = Path(args.checkpoint) if args.checkpoint else None

    print("[INFO] Cargando datos...")
    feature_dicts = _load(parquet_p)
    if not feature_dicts:
        print("ERROR: No hay datos de entrenamiento disponibles.")
        return 1

    seasons = sorted({f.get("season", "") for f in feature_dicts if f.get("season")})
    print(f"[INFO] {len(feature_dicts)} partidos | Temporadas: {seasons}")

    # Configuración del ensemble
    cfg_path = Path("checkpoints/ensemble/config.json")
    config = {}
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))

    all_results = []

    if args.strategy in ("season", "both"):
        r = evaluate_season_split(feature_dicts, args.target_season, ckpt_p, config)
        all_results.extend(r)

    if args.strategy in ("rolling", "both"):
        r = evaluate_rolling(
            feature_dicts,
            args.target_season,
            args.train_jornadas,
            args.eval_jornadas,
            ckpt_p,
            config,
        )
        all_results.extend(r)

    _print_summary(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
