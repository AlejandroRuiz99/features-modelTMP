#!/usr/bin/env python3
"""
train.py — Entrena el ensemble FoulPredictionEnsemble desde el Parquet de training.

Pipeline:
  1. Carga training.parquet
  2. Split temporal: train=2023-24, tune=2024-25, test=2025-26
  3. Computa team averages desde datos de train
  4. Fit del ensemble (NB + NegBin + ANFIS + Gating)
  5. Grid search de hiperparámetros post-hoc (prior_mix, min_weight, variance_scale) en tune set
  6. Aplica los mejores hiperparámetros y calibración isotónica en tune set
  7. Guarda checkpoint en prediction_models/checkpoints/ensemble/

Uso:
  python scripts/train.py
  python scripts/train.py --data prediction_models/data/training.parquet --tune-season 2024-25 --test-season 2025-26
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "prediction_models"))
sys.path.insert(0, str(_ROOT / "features_generator"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA = _ROOT / "prediction_models" / "data" / "training.parquet"
DEFAULT_CHECKPOINT = _ROOT / "prediction_models" / "checkpoints" / "ensemble"
DEFAULT_CONFIG = DEFAULT_CHECKPOINT / "config.json"


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    train_seasons: list[str],
    tune_season: str,
    test_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into 3 non-overlapping temporal partitions.

    When ``tune_season`` and ``test_season`` are the same string, the rows
    belonging to that season are divided chronologically: the first half goes
    to *tune* and the second half goes to *test*.  This ensures all three
    splits are non-overlapping and their row counts sum to ``len(df)``.

    A ``UserWarning`` is emitted for each requested season that is absent from
    ``df["season"]``; the corresponding split is returned as an empty DataFrame.

    Args:
        df: Full DataFrame sorted (or at least grouped) by date/season.
        train_seasons: List of season strings to include in the train split.
        tune_season: Season string for the tune split.
        test_season: Season string for the test split.

    Returns:
        A 3-tuple ``(train_df, tune_df, test_df)``.
    """
    available_seasons: set[str] = set(df["season"].unique())

    # Warn about any missing seasons
    all_requested = set(train_seasons) | {tune_season, test_season}
    for season in sorted(all_requested - available_seasons):
        warnings.warn(
            f"Season '{season}' not found in the DataFrame "
            f"(available: {sorted(available_seasons)}). "
            "The corresponding split will be empty.",
            UserWarning,
            stacklevel=2,
        )

    # Train split: union of all train_seasons rows
    train_mask = df["season"].isin(train_seasons)
    train_df = df[train_mask].copy()

    if tune_season == test_season:
        # Same season → split chronologically 50/50
        season_df = df[df["season"] == tune_season].copy()
        if len(season_df) == 0:
            tune_df: pd.DataFrame = season_df.iloc[:0].copy()
            test_df: pd.DataFrame = season_df.iloc[:0].copy()
        else:
            mid = len(season_df) // 2
            tune_df = season_df.iloc[:mid].copy()
            test_df = season_df.iloc[mid:].copy()
    else:
        tune_df = df[df["season"] == tune_season].copy()
        test_df = df[df["season"] == test_season].copy()

    logger.info(
        "Temporal split — train: %d rows | tune: %d rows | test: %d rows",
        len(train_df),
        len(tune_df),
        len(test_df),
    )

    return train_df, tune_df, test_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _team_averages(matches: list[dict]) -> tuple[dict, dict, dict]:
    """Computa promedios por equipo para NB desde los feature dicts."""
    committed: dict[str, list] = {}
    suffered: dict[str, list] = {}
    rank: dict[str, list] = {}

    for m in matches:
        for side, opp in [("home", "away"), ("away", "home")]:
            team = m.get(f"{side}_team", "")
            if not team:
                continue
            committed.setdefault(team, []).append(
                float(m.get(f"{side}_fouls_committed_avg", 12.0))
            )
            suffered.setdefault(team, []).append(
                float(m.get(f"{side}_fouls_suffered_avg", 12.0))
            )
            rank.setdefault(team, []).append(float(m.get(f"{side}_rank_hist", 10.0)))

    return (
        {t: float(np.mean(v)) for t, v in committed.items()},
        {t: float(np.mean(v)) for t, v in suffered.items()},
        {t: float(np.mean(v)) for t, v in rank.items()},
    )


def _mae(predictions: list, actuals: list[float]) -> float:
    return float(
        np.mean([abs(p.expected_fouls - a) for p, a in zip(predictions, actuals)])
    )


def _nll(predictions: list, actuals: list[float]) -> float:
    nlls = []
    for pred, actual in zip(predictions, actuals):
        k = max(0, min(int(actual), 60))
        probs = (
            np.array(pred.pmf_total.probs)
            if hasattr(pred.pmf_total, "probs")
            else np.array(pred.pmf_total._probs)
        )
        p = float(probs[k])
        nlls.append(-np.log(max(p, 1e-10)))
    return float(np.mean(nlls))


def _evaluate(ensemble, matches: list[dict]) -> dict:
    predictions, actuals = [], []
    for m in matches:
        ft = m.get("fouls_total")
        if ft is None:
            continue
        try:
            predictions.append(ensemble.predict(m))
            actuals.append(float(ft))
        except Exception:
            pass
    if not predictions:
        return {"mae": float("nan"), "nll": float("nan"), "n": 0, "bias": float("nan")}
    mae = _mae(predictions, actuals)
    nll = _nll(predictions, actuals)
    bias = float(np.mean([p.expected_fouls - a for p, a in zip(predictions, actuals)]))
    return {
        "mae": round(mae, 4),
        "nll": round(nll, 4),
        "n": len(predictions),
        "bias": round(bias, 4),
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def _grid_search(ensemble, test_matches: list[dict], checkpoint_dir: Path) -> dict:
    """Busca los mejores prior_mix, min_weight, variance_scale en el test set."""
    prior_mix_values = [0.10, 0.15, 0.20, 0.25, 0.35]
    min_weight_values = [0.01, 0.02, 0.03, 0.05]
    variance_scale_values = [0.30, 0.40, 0.50, 0.60, 0.75]

    results = []
    total = len(prior_mix_values) * len(min_weight_values) * len(variance_scale_values)
    i = 0

    for prior_mix in prior_mix_values:
        for min_weight in min_weight_values:
            for variance_scale in variance_scale_values:
                i += 1
                ensemble.weighter.prior_mix = prior_mix
                ensemble.weighter.min_weight = min_weight
                ensemble._variance_posthoc_scale = variance_scale

                metrics = _evaluate(ensemble, test_matches)
                entry = {
                    "prior_mix": prior_mix,
                    "min_weight": min_weight,
                    "variance_scale": variance_scale,
                    **metrics,
                }
                results.append(entry)

                if i % 12 == 0 or i == total:
                    logger.info("  Grid search %d/%d completado", i, total)

    # Composite selection: normalise MAE and NLL then combine
    mean_mae = sum(r["mae"] for r in results) / len(results) if results else 1.0
    mean_nll = sum(r["nll"] for r in results) / len(results) if results else 1.0
    for r in results:
        mae_norm = r["mae"] / mean_mae if mean_mae != 0 else 1.0
        nll_norm = r["nll"] / mean_nll if mean_nll != 0 else 1.0
        r["composite"] = round(0.6 * mae_norm + 0.4 * nll_norm, 6)
    best = min(results, key=lambda r: r["composite"])

    # Guardar resultados del grid search
    results.sort(key=lambda r: r["composite"])
    grid_path = checkpoint_dir / "tuning_magic_constants.json"
    with open(grid_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(
        "Grid search guardado: %s (%d combinaciones, n=%d)",
        grid_path,
        len(results),
        test_matches and results[0].get("n", 0),
    )

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar ensemble foultsPredictor")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--tune-season",
        default="2024-25",
        help="Temporada usada como tune set (grid search, calibración, gating)",
    )
    p.add_argument(
        "--test-season",
        default="2025-26",
        help="Temporada usada como test set (evaluación final únicamente)",
    )
    p.add_argument(
        "--no-team-models", action="store_true", help="Omitir TeamFoulRegressor"
    )
    p.add_argument("--no-grid-search", action="store_true")
    return p.parse_args()


def main() -> None:
    from src.models.ensemble import FoulPredictionEnsemble

    args = _parse_args()
    data_path = Path(args.data)
    checkpoint_dir = Path(args.checkpoint)
    config_path = Path(args.config)

    if not data_path.exists():
        logger.error("Parquet no encontrado: %s", data_path)
        sys.exit(1)

    # -- Cargar config --
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        logger.info("Config cargado desde %s", config_path)

    # -- Cargar y splitear datos --
    logger.info("Cargando parquet: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info(
        "  %d filas x %d cols | temporadas: %s",
        len(df),
        len(df.columns),
        sorted(df["season"].unique()),
    )

    # 3-way temporal split: train / tune / test
    # AD-6: train=seasons before tune_season, tune=tune_season, test=test_season
    available_seasons = sorted(df["season"].unique())
    train_seasons = [s for s in available_seasons if s < args.tune_season]

    train_df, tune_df, test_df = temporal_split(
        df,
        train_seasons=train_seasons,
        tune_season=args.tune_season,
        test_season=args.test_season,
    )

    train_matches = train_df.to_dict("records")
    tune_matches = tune_df.to_dict("records")
    test_matches = test_df.to_dict("records")

    # -- Promedios por equipo --
    team_committed, team_suffered, team_rank = _team_averages(train_matches)
    logger.info("  %d equipos con promedios calculados", len(team_committed))

    # -- Entrenar --
    logger.info("Iniciando entrenamiento...")
    t0 = time.time()
    ensemble = FoulPredictionEnsemble(config)
    ensemble.fit(
        train_matches,
        team_avg_committed=team_committed,
        team_avg_suffered=team_suffered,
        team_avg_rank=team_rank,
        fit_team_models=not args.no_team_models,
        gating_matches=tune_matches,  # gating out-of-sample on TUNE set (AD-6)
    )
    elapsed = time.time() - t0
    logger.info("Entrenamiento completado en %.1fs", elapsed)

    # -- Evaluar baseline (en tune set) --
    logger.info("Evaluando en tune set (antes de grid search y calibración)...")
    baseline = _evaluate(ensemble, tune_matches)
    logger.info(
        "  MAE=%.4f | NLL=%.4f | bias=%.4f | n=%d",
        baseline["mae"],
        baseline["nll"],
        baseline["bias"],
        baseline["n"],
    )

    # -- Grid search en tune set (no tocar test set) --
    best_params = {}
    if not args.no_grid_search and len(tune_matches) >= 30:
        logger.info("Grid search de hiperparámetros (%d combinaciones)...", 5 * 4 * 5)
        best_params = _grid_search(
            ensemble, tune_matches, checkpoint_dir
        )  # tune, not test (AD-6)
        logger.info(
            "Mejores parámetros: prior_mix=%.2f | min_weight=%.2f | variance_scale=%.2f | MAE=%.4f | NLL=%.4f | composite=%.6f",
            best_params["prior_mix"],
            best_params["min_weight"],
            best_params["variance_scale"],
            best_params["mae"],
            best_params["nll"],
            best_params.get("composite", float("nan")),
        )
        # Aplicar los mejores
        ensemble.weighter.prior_mix = best_params["prior_mix"]
        ensemble.weighter.min_weight = best_params["min_weight"]
        ensemble._variance_posthoc_scale = best_params["variance_scale"]
    else:
        logger.info("Grid search omitido (tune set < 30 o --no-grid-search).")

    # -- Calibración isotónica en tune set (no tocar test set) --
    logger.info("Calibrando con datos de tune (%d partidos)...", len(tune_matches))
    ensemble.calibrate_fit(tune_matches)  # tune, not test (AD-6)

    # -- Evaluar final en test set (única vez que se usa test set) --
    logger.info("Evaluación final en TEST set (post grid search + calibración)...")
    final = _evaluate(ensemble, test_matches)  # test set only here
    logger.info(
        "  MAE=%.4f | NLL=%.4f | bias=%.4f | n=%d",
        final["mae"],
        final["nll"],
        final["bias"],
        final["n"],
    )

    # -- Guardar checkpoint --
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(checkpoint_dir)
    logger.info("Checkpoint guardado en: %s", checkpoint_dir)

    # -- Resumen --
    print("\n" + "=" * 60)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 60)
    print(f"Train:  {len(train_matches)} partidos ({train_seasons})")
    print(f"Tune:   {len(tune_matches)} partidos ({args.tune_season!r})")
    print(f"Test:   {len(test_matches)} partidos ({args.test_season!r})")
    print(f"Tiempo entrenamiento: {elapsed:.1f}s")
    print(
        f"\nBaseline (tune) — MAE={baseline['mae']:.4f} | NLL={baseline['nll']:.4f} | bias={baseline['bias']:+.4f}"
    )
    print(
        f"Final (test)    — MAE={final['mae']:.4f} | NLL={final['nll']:.4f} | bias={final['bias']:+.4f}"
    )
    if best_params:
        print("\nMejores hiperparámetros:")
        print(
            f"  prior_mix={best_params['prior_mix']} | min_weight={best_params['min_weight']} | variance_scale={best_params['variance_scale']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
