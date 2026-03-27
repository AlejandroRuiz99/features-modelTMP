"""
Tune de constantes sensibles con walk-forward sobre las ultimas jornadas.

Prioriza los parametros:
  - gating_network.prior_mix
  - gating_network.min_weight
  - reconciliation.total_variance_scale
  - anfis.match_intensity_weights
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import load_training_parquet, build_team_stats
from src.models.ensemble import FoulPredictionEnsemble


def _season_key(s: str) -> int:
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return 0


def _base_config() -> dict:
    cfg_path = Path("checkpoints/ensemble/config.json")
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
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
            "output_center": 26.0,
            "output_scale": 15.0,
            "output_tanh_temp": 5.0,
            "match_intensity_weights": [0.25, 0.20, 0.25, 0.15, 0.15],
            "play_style_weights": [0.40, 0.35, 0.25],
        },
        "gating_network": {
            "hidden_dims": [48, 24],
            "dropout": 0.15,
            "temperature": 1.0,
            "lr": 0.001,
            "min_weight": 0.15,
            "prior_mix": 0.35,
        },
        "reconciliation": {"total_variance_scale": 0.6},
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mae(preds: list[float], reals: list[float]) -> float:
    if not preds:
        return float("nan")
    return float(np.mean(np.abs(np.array(preds) - np.array(reals))))


def _bias(preds: list[float], reals: list[float]) -> float:
    if not preds:
        return float("nan")
    return float(np.mean(np.array(preds) - np.array(reals)))


def _build_config(
    base: dict,
    prior_mix: float,
    min_weight: float,
    variance_scale: float,
    intensity_weights: list[float],
) -> dict:
    cfg = copy.deepcopy(base)
    cfg.setdefault("gating_network", {})
    cfg.setdefault("reconciliation", {})
    cfg.setdefault("anfis", {})
    cfg["gating_network"]["prior_mix"] = float(prior_mix)
    cfg["gating_network"]["min_weight"] = float(min_weight)
    cfg["reconciliation"]["total_variance_scale"] = float(variance_scale)
    cfg["anfis"]["match_intensity_weights"] = [float(x) for x in intensity_weights]
    return cfg


def _eval_walk_forward(
    feature_dicts: list[dict],
    target_season: str,
    eval_jornadas: list[int],
    cfg: dict,
    seed: int,
) -> tuple[float, float, int]:
    target_key = _season_key(target_season)
    preds: list[float] = []
    reals: list[float] = []
    n_pred = 0

    for jornada in eval_jornadas:
        train_feats = [
            f for f in feature_dicts
            if _season_key(f.get("season", "")) < target_key
            or (f.get("season") == target_season and int(f.get("matchday", 0)) < jornada)
        ]
        eval_feats = [
            f for f in feature_dicts
            if f.get("season") == target_season and int(f.get("matchday", 0)) == jornada
        ]
        if len(train_feats) < 20 or not eval_feats:
            continue

        _set_seed(seed + jornada)
        avg_c, avg_s, avg_r = build_team_stats(train_feats)
        ensemble = FoulPredictionEnsemble(config=cfg)
        ensemble.fit(
            matches=train_feats,
            team_avg_committed=avg_c,
            team_avg_suffered=avg_s,
            team_avg_rank=avg_r,
            fit_team_models=False,
            gating_matches=None,
        )

        for m in eval_feats:
            try:
                p = ensemble.predict(m).expected_fouls
                preds.append(float(p))
                reals.append(float(m["fouls_total"]))
                n_pred += 1
            except Exception:
                continue

    return _mae(preds, reals), _bias(preds, reals), n_pred


def tune(
    parquet_path: Path,
    target_season: str,
    last_n_jornadas: int,
    max_combos: int,
    seed: int,
) -> list[dict]:
    feats = load_training_parquet(parquet_path)
    target = [f for f in feats if f.get("season") == target_season]
    jornadas = sorted({int(f.get("matchday", 0)) for f in target if f.get("matchday") is not None})
    if not jornadas:
        raise ValueError(f"No hay datos de {target_season}")
    eval_jornadas = jornadas[-last_n_jornadas:]

    base = _base_config()
    intensity_candidates = [
        [0.25, 0.20, 0.25, 0.15, 0.15],  # baseline
        [0.20, 0.20, 0.30, 0.15, 0.15],  # más urgencia
        [0.30, 0.20, 0.20, 0.15, 0.15],  # más derby
        [0.20, 0.25, 0.25, 0.15, 0.15],  # más ranking
    ]
    combos = list(itertools.product(
        [0.25, 0.35, 0.45],   # prior_mix
        [0.10, 0.15, 0.20],   # min_weight
        [0.45, 0.60, 0.75],   # variance_scale
        range(len(intensity_candidates)),
    ))
    if max_combos > 0:
        combos = combos[:max_combos]

    print(f"[INFO] Eval walk-forward jornadas {eval_jornadas} | combos={len(combos)}")
    results: list[dict] = []
    for idx, (pmix, minw, vscale, iw_idx) in enumerate(combos, 1):
        iw = intensity_candidates[iw_idx]
        cfg = _build_config(base, pmix, minw, vscale, iw)
        mae, bias, n_pred = _eval_walk_forward(
            feats, target_season, eval_jornadas, cfg, seed=seed
        )
        row = {
            "prior_mix": pmix,
            "min_weight": minw,
            "variance_scale": vscale,
            "match_intensity_weights": iw,
            "mae": mae,
            "bias": bias,
            "n_pred": n_pred,
        }
        results.append(row)
        print(
            f"[{idx:03d}/{len(combos):03d}] "
            f"mae={mae:.3f} bias={bias:+.3f} "
            f"pmix={pmix:.2f} minw={minw:.2f} vscale={vscale:.2f} iw={iw}"
        )

    results.sort(key=lambda r: (r["mae"], abs(r["bias"])))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune de constantes mágicas (walk-forward)")
    parser.add_argument("--parquet", type=str, default="data/training.parquet")
    parser.add_argument("--target-season", type=str, default="2025-26")
    parser.add_argument("--last-jornadas", type=int, default=10)
    parser.add_argument("--max-combos", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = tune(
        parquet_path=Path(args.parquet),
        target_season=args.target_season,
        last_n_jornadas=args.last_jornadas,
        max_combos=args.max_combos,
        seed=args.seed,
    )
    print("\n=== TOP 5 CONFIGS ===")
    for i, r in enumerate(results[:5], 1):
        print(
            f"{i}. MAE={r['mae']:.3f} bias={r['bias']:+.3f} "
            f"prior_mix={r['prior_mix']:.2f} min_weight={r['min_weight']:.2f} "
            f"variance_scale={r['variance_scale']:.2f} iw={r['match_intensity_weights']}"
        )

    out = Path("checkpoints/ensemble/tuning_magic_constants.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[OK] Guardado ranking completo en {out}")
    return 0


if __name__ == "__main__":
    # Evita buffering de salida en algunas consolas/entornos.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    raise SystemExit(main())

