"""Unit tests for prediction_models/src/models/naive_bayes.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "prediction_models"))

import numpy as np
import pytest

from src.models.naive_bayes import FoulDiscretizer, NaiveBayesFoulPredictor


# ---------------------------------------------------------------------------
# Helpers — build minimal training data
# ---------------------------------------------------------------------------

def _make_matches(n: int = 40) -> list[dict]:
    """Generate synthetic match dicts that satisfy all required keys."""
    rng = np.random.default_rng(42)
    teams = [f"Team{i}" for i in range(20)]
    matches = []
    for i in range(n):
        home = teams[i % 20]
        away = teams[(i + 1) % 20]
        fouls_total = int(rng.integers(18, 45))
        matches.append(
            {
                "home_team": home,
                "away_team": away,
                "fouls_total": fouls_total,
                "home_fouls_committed_avg": float(rng.uniform(10, 16)),
                "away_fouls_committed_avg": float(rng.uniform(10, 16)),
                "home_fouls_suffered_avg": float(rng.uniform(10, 16)),
                "away_fouls_suffered_avg": float(rng.uniform(10, 16)),
                "home_rank_hist": float(rng.uniform(1, 20)),
                "away_rank_hist": float(rng.uniform(1, 20)),
                "home_rank_curr": float(rng.uniform(1, 20)),
                "away_rank_curr": float(rng.uniform(1, 20)),
                "xfouls_home": float(rng.uniform(10, 16)),
                "xfouls_away": float(rng.uniform(10, 16)),
                "aggressiveness_volume_home": float(rng.uniform(0.3, 0.8)),
                "aggressiveness_volume_away": float(rng.uniform(0.3, 0.8)),
                "ref_home_delta": float(rng.uniform(-2, 2)),
                "ref_away_delta": float(rng.uniform(-2, 2)),
                "referee_mode": int(rng.integers(0, 2)),
            }
        )
    return matches


def _make_team_avgs(matches: list[dict]) -> tuple[dict, dict, dict]:
    committed: dict[str, list] = {}
    suffered: dict[str, list] = {}
    rank: dict[str, list] = {}
    for m in matches:
        for side in ("home", "away"):
            t = m[f"{side}_team"]
            committed.setdefault(t, []).append(m[f"{side}_fouls_committed_avg"])
            suffered.setdefault(t, []).append(m[f"{side}_fouls_suffered_avg"])
            rank.setdefault(t, []).append(m[f"{side}_rank_hist"])
    return (
        {t: float(np.mean(v)) for t, v in committed.items()},
        {t: float(np.mean(v)) for t, v in suffered.items()},
        {t: float(np.mean(v)) for t, v in rank.items()},
    )


def _make_match() -> dict:
    """Single match dict for prediction."""
    return {
        "home_team": "Team0",
        "away_team": "Team1",
        "fouls_total": 28,
        "home_fouls_committed_avg": 13.0,
        "away_fouls_committed_avg": 13.0,
        "home_fouls_suffered_avg": 12.0,
        "away_fouls_suffered_avg": 12.0,
        "home_rank_hist": 10.0,
        "away_rank_hist": 10.0,
        "home_rank_curr": 10.0,
        "away_rank_curr": 10.0,
        "xfouls_home": 12.5,
        "xfouls_away": 12.5,
        "aggressiveness_volume_home": 0.5,
        "aggressiveness_volume_away": 0.5,
        "ref_home_delta": 0.0,
        "ref_away_delta": 0.0,
        "referee_mode": 0,
        "referee_strict_prob": 0.5,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFoulDiscretizerDefaults:
    def test_default_produces_8_intervals(self) -> None:
        """FoulDiscretizer() with no args → after fit: n_classes == 8, len(breakpoints) == 9."""
        disc = FoulDiscretizer()
        data = np.arange(5, 50, dtype=float)
        disc.fit(data)
        assert disc.n_classes == 8
        assert len(disc.breakpoints) == 9

    def test_custom_percentiles_respected(self) -> None:
        """FoulDiscretizer(percentiles=[25, 50, 75]) → n_classes == 4."""
        disc = FoulDiscretizer(percentiles=[25, 50, 75])
        data = np.arange(5, 50, dtype=float)
        disc.fit(data)
        assert disc.n_classes == 4


class TestNaiveBayesFoulPredictor:
    def test_naive_bayes_propagates_foul_percentiles(self) -> None:
        """NaiveBayesFoulPredictor(foul_percentiles=[10,25,50,75,90]) after fit → foul_discretizer.n_classes == 6."""
        predictor = NaiveBayesFoulPredictor(foul_percentiles=[10, 25, 50, 75, 90])
        matches = _make_matches(40)
        avg_committed, avg_suffered, avg_rank = _make_team_avgs(matches)
        predictor.fit(matches, avg_committed, avg_suffered, avg_rank)
        assert predictor.foul_discretizer.n_classes == 6

    def test_predict_interval_probs_shape_matches_n_classes(self) -> None:
        """Default NB after fit → predict_interval_probs(match).shape == (8,) and sum ≈ 1.0."""
        predictor = NaiveBayesFoulPredictor()
        matches = _make_matches(40)
        avg_committed, avg_suffered, avg_rank = _make_team_avgs(matches)
        predictor.fit(matches, avg_committed, avg_suffered, avg_rank)
        match = _make_match()
        probs = predictor.predict_interval_probs(match)
        assert probs.shape == (8,)
        assert abs(probs.sum() - 1.0) < 1e-6
