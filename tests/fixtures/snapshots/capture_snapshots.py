"""
capture_snapshots.py — Regenerate regression snapshots for the overlay test suite.

Usage:
    python tests/fixtures/snapshots/capture_snapshots.py

This script loads three feature fixture dicts directly (no Supabase, no network),
runs them through the ensemble, and writes JSON snapshots to this directory.
The snapshots are the regression contract:
  "no narrative supplied → output byte-identical to snapshot"

Run this script to regenerate snapshots whenever the ensemble model is updated
(retrain). Commit the regenerated snapshots after reviewing the diffs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = ROOT / "prediction_models"
sys.path.insert(0, str(PRED_DIR))

from src.models.ensemble import FoulPredictionEnsemble  # noqa: E402

CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"
SNAPSHOT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Feature fixtures — three synthetic match snapshots (no Supabase required)
# ---------------------------------------------------------------------------

# Fixture 1: Espanyol vs Levante J32 2026-04-27 (from features_dump.json)
FIXTURE_ESPANYOL_LEVANTE: dict = {
    "home_team": "Espanol",
    "away_team": "Levante",
    "referee": "Busquets Ferrer",
    "matchday": 32,
    "season": "2025-26",
    "date": "2026-04-27",
    "home_fouls_committed_avg": 13.7,
    "home_fouls_suffered_avg": 12.5,
    "away_fouls_committed_avg": 12.7,
    "away_fouls_suffered_avg": 13.5,
    "home_fouls_committed_curr": 15.8,
    "away_fouls_committed_curr": 14.8,
    "home_shots_curr": 11.7,
    "away_shots_curr": 12.2,
    "home_corners_curr": 4.5,
    "away_corners_curr": 4.1,
    "home_yellows_avg": 2.43,
    "away_yellows_avg": 2.28,
    "home_reds_avg": 0.102,
    "away_reds_avg": 0.137,
    "fouls_provoked_home": 12.5,
    "fouls_provoked_away": 13.5,
    "home_rank_hist": 12.0,
    "away_rank_hist": 19.0,
    "home_rank_curr": 12,
    "away_rank_curr": 19,
    "rank_diff_norm": 0.3684,
    "home_xg": 1.52,
    "away_xg": 1.12,
    "xg_diff": 0.3999999999999999,
    "home_possession": 0.502,
    "away_possession": 0.498,
    "xfouls_home": 14.6,
    "xfouls_away": 13.0,
    "xfouls_factor_home": 0.9935,
    "xfouls_factor_away": 1.0307,
    "aggressiveness_volume_home": 0.994,
    "aggressiveness_volume_away": 0.8853,
    "aggressiveness_norm_total": 0.9397,
    "forma_fouls_home": 15.8,
    "forma_fouls_away": 14.8,
    "momentum_home": 0.067,
    "momentum_away": 0.667,
    "season_phase": 0.8421,
    "is_derby": False,
    "urgency_home": 0.41,
    "urgency_away": 0.812,
    "fatigue_home": 0.801,
    "fatigue_away": 0.801,
    "days_rest_home": 1.0,
    "days_rest_away": 1.0,
    "referee_mu_permisivo": 20.93,
    "referee_mu_estricto": 30.44,
    "referee_sigma_permisivo": 3.51,
    "referee_sigma_estricto": 3.31,
    "referee_peso_estricto": 0.386,
    "referee_n_partidos": 52,
    "referee_clean_avg": 23.9,
    "ref_home_delta": -0.19,
    "ref_away_delta": 0.0,
    "ref_pair_delta_sum": -0.19,
    "ref_pair_samples": 6.0,
    "has_market_odds": False,
    "market_home_win_prob": 0.3773,
    "market_draw_prob": 0.2858,
    "market_away_win_prob": 0.3369,
    "market_favorite_prob": 0.3773,
    "market_balance": 0.9596,
    "market_entropy": 1.0923,
    "market_ou25_over_prob": 0.5,
    "market_ou25_under_prob": 0.5,
    "foul_market_prob_over": 0.5,
    "foul_market_implied_mean": 24.5,
    "intensidad_esperada": "alta",
    "riesgo_disciplinario": "alto",
    "pace_index_curr": 32.8,
    "h2h_faltas_media": 19.0,
    "h2h_partidos": 1,
    "foul_market_local_implied_mean": 12.25,
    "foul_market_vis_implied_mean": 12.25,
}

# Fixture 2: Real Madrid vs Atletico Madrid — mid-table vs top (synthetic)
FIXTURE_MADRID_ATLETICO: dict = {
    "home_team": "Real Madrid",
    "away_team": "Atletico Madrid",
    "referee": "Medié Jiménez",
    "matchday": 29,
    "season": "2025-26",
    "date": "2026-03-22",
    "home_fouls_committed_avg": 11.2,
    "home_fouls_suffered_avg": 13.1,
    "away_fouls_committed_avg": 14.5,
    "away_fouls_suffered_avg": 11.8,
    "home_fouls_committed_curr": 12.0,
    "away_fouls_committed_curr": 15.2,
    "home_shots_curr": 14.3,
    "away_shots_curr": 10.5,
    "home_corners_curr": 5.8,
    "away_corners_curr": 3.9,
    "home_yellows_avg": 1.95,
    "away_yellows_avg": 2.85,
    "home_reds_avg": 0.05,
    "away_reds_avg": 0.08,
    "fouls_provoked_home": 13.1,
    "fouls_provoked_away": 11.8,
    "home_rank_hist": 1.0,
    "away_rank_hist": 3.0,
    "home_rank_curr": 1,
    "away_rank_curr": 3,
    "rank_diff_norm": 0.1053,
    "home_xg": 2.15,
    "away_xg": 1.45,
    "xg_diff": 0.7,
    "home_possession": 0.58,
    "away_possession": 0.42,
    "xfouls_home": 12.5,
    "xfouls_away": 14.8,
    "xfouls_factor_home": 0.89,
    "xfouls_factor_away": 1.06,
    "aggressiveness_volume_home": 0.75,
    "aggressiveness_volume_away": 1.15,
    "aggressiveness_norm_total": 0.95,
    "forma_fouls_home": 12.0,
    "forma_fouls_away": 15.2,
    "momentum_home": 0.82,
    "momentum_away": 0.45,
    "season_phase": 0.7368,
    "is_derby": True,
    "urgency_home": 0.25,
    "urgency_away": 0.35,
    "fatigue_home": 0.9,
    "fatigue_away": 0.85,
    "days_rest_home": 7.0,
    "days_rest_away": 7.0,
    "referee_mu_permisivo": 22.1,
    "referee_mu_estricto": 29.8,
    "referee_sigma_permisivo": 3.8,
    "referee_sigma_estricto": 3.2,
    "referee_peso_estricto": 0.42,
    "referee_n_partidos": 38,
    "referee_clean_avg": 25.1,
    "ref_home_delta": 0.12,
    "ref_away_delta": 0.22,
    "ref_pair_delta_sum": 0.34,
    "ref_pair_samples": 3.0,
    "has_market_odds": False,
    "market_home_win_prob": 0.52,
    "market_draw_prob": 0.26,
    "market_away_win_prob": 0.22,
    "market_favorite_prob": 0.52,
    "market_balance": 0.95,
    "market_entropy": 1.03,
    "market_ou25_over_prob": 0.5,
    "market_ou25_under_prob": 0.5,
    "foul_market_prob_over": 0.5,
    "foul_market_implied_mean": 26.5,
    "intensidad_esperada": "alta",
    "riesgo_disciplinario": "medio",
    "pace_index_curr": 29.5,
    "h2h_faltas_media": 27.5,
    "h2h_partidos": 8,
    "foul_market_local_implied_mean": 12.5,
    "foul_market_vis_implied_mean": 14.0,
}

# Fixture 3: Villarreal vs Osasuna — mid-table (synthetic)
FIXTURE_VILLARREAL_OSASUNA: dict = {
    "home_team": "Villarreal",
    "away_team": "Osasuna",
    "referee": "Munuera Montero",
    "matchday": 25,
    "season": "2025-26",
    "date": "2026-02-15",
    "home_fouls_committed_avg": 12.4,
    "home_fouls_suffered_avg": 11.9,
    "away_fouls_committed_avg": 13.8,
    "away_fouls_suffered_avg": 12.2,
    "home_fouls_committed_curr": 13.1,
    "away_fouls_committed_curr": 14.3,
    "home_shots_curr": 12.5,
    "away_shots_curr": 9.8,
    "home_corners_curr": 4.9,
    "away_corners_curr": 3.5,
    "home_yellows_avg": 2.1,
    "away_yellows_avg": 2.6,
    "home_reds_avg": 0.07,
    "away_reds_avg": 0.11,
    "fouls_provoked_home": 11.9,
    "fouls_provoked_away": 12.2,
    "home_rank_hist": 7.0,
    "away_rank_hist": 11.0,
    "home_rank_curr": 7,
    "away_rank_curr": 11,
    "rank_diff_norm": 0.2105,
    "home_xg": 1.72,
    "away_xg": 1.05,
    "xg_diff": 0.67,
    "home_possession": 0.54,
    "away_possession": 0.46,
    "xfouls_home": 13.0,
    "xfouls_away": 14.2,
    "xfouls_factor_home": 0.96,
    "xfouls_factor_away": 1.08,
    "aggressiveness_volume_home": 0.88,
    "aggressiveness_volume_away": 1.05,
    "aggressiveness_norm_total": 0.965,
    "forma_fouls_home": 13.1,
    "forma_fouls_away": 14.3,
    "momentum_home": 0.55,
    "momentum_away": 0.33,
    "season_phase": 0.6316,
    "is_derby": False,
    "urgency_home": 0.30,
    "urgency_away": 0.45,
    "fatigue_home": 0.95,
    "fatigue_away": 0.92,
    "days_rest_home": 7.0,
    "days_rest_away": 7.0,
    "referee_mu_permisivo": 21.5,
    "referee_mu_estricto": 31.2,
    "referee_sigma_permisivo": 3.2,
    "referee_sigma_estricto": 3.5,
    "referee_peso_estricto": 0.45,
    "referee_n_partidos": 65,
    "referee_clean_avg": 25.5,
    "ref_home_delta": 0.05,
    "ref_away_delta": -0.10,
    "ref_pair_delta_sum": -0.05,
    "ref_pair_samples": 2.0,
    "has_market_odds": False,
    "market_home_win_prob": 0.45,
    "market_draw_prob": 0.28,
    "market_away_win_prob": 0.27,
    "market_favorite_prob": 0.45,
    "market_balance": 0.97,
    "market_entropy": 1.08,
    "market_ou25_over_prob": 0.5,
    "market_ou25_under_prob": 0.5,
    "foul_market_prob_over": 0.5,
    "foul_market_implied_mean": 25.5,
    "intensidad_esperada": "media",
    "riesgo_disciplinario": "medio",
    "pace_index_curr": 27.8,
    "h2h_faltas_media": 25.0,
    "h2h_partidos": 5,
    "foul_market_local_implied_mean": 13.0,
    "foul_market_vis_implied_mean": 12.5,
}

FIXTURES = [
    ("espanyol_levante_j32", FIXTURE_ESPANYOL_LEVANTE),
    ("realmadrid_atletico_j29", FIXTURE_MADRID_ATLETICO),
    ("villarreal_osasuna_j25", FIXTURE_VILLARREAL_OSASUNA),
]


def _load_ensemble() -> FoulPredictionEnsemble:
    config_path = CHECKPOINT_DIR / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(CHECKPOINT_DIR)
    return ensemble


def _prediction_to_snapshot(ensemble: FoulPredictionEnsemble, feat: dict) -> dict:
    """Run ensemble prediction and return a deterministic JSON-serializable snapshot."""
    pred = ensemble.predict(feat)
    team_pred = (
        ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True) or {}
    )
    return {
        "match": f"{feat['home_team']} vs {feat['away_team']}",
        "date": feat.get("date", ""),
        "jornada": feat.get("matchday", 0),
        "referee": feat.get("referee", ""),
        "expected_fouls": float(pred.expected_fouls),
        "referee_strict_prob": float(pred.referee_strict_prob),
        "weights": [float(w) for w in pred.weights],
        "home_expected": float(team_pred.get("home_expected", 0.0)),
        "away_expected": float(team_pred.get("away_expected", 0.0)),
        "total_expected": float(team_pred.get("total_expected", pred.expected_fouls)),
        "reconciled": bool(team_pred.get("reconciled", False)),
        "over_under": {
            str(k): {"over": float(v[0]), "under": float(v[1])}
            for k, v in (pred.over_under or {}).items()
        },
    }


def main() -> None:
    print(f"Loading ensemble from {CHECKPOINT_DIR}...")
    ensemble = _load_ensemble()
    print("Ensemble loaded.")

    for name, feat in FIXTURES:
        snap = _prediction_to_snapshot(ensemble, feat)
        out_path = SNAPSHOT_DIR / f"{name}.json"
        out_path.write_text(
            json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"  Saved: {out_path.name}  (expected_fouls={snap['expected_fouls']:.4f})"
        )

    print(f"\nSnapshots saved to {SNAPSHOT_DIR}")
    print("Commit these files to lock the regression contract.")


if __name__ == "__main__":
    main()
