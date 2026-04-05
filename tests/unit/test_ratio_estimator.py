"""Unit tests for HomeFoulRatioEstimator (TDD — tests written before implementation).

These tests WILL FAIL until T-07/T-08 are implemented. That is correct for TDD.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "prediction_models"))

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Shared fixtures — minimal match data
# ---------------------------------------------------------------------------

def _make_matches(n: int = 40) -> list[dict]:
    """Generate synthetic match dicts with all keys required by HomeFoulRatioEstimator."""
    rng = np.random.default_rng(42)
    matches = []
    for i in range(n):
        fouls_home = int(rng.integers(8, 20))
        fouls_away = int(rng.integers(8, 20))
        fouls_total = fouls_home + fouls_away
        matches.append(
            {
                "home_team": f"Team{i % 10}",
                "away_team": f"Team{(i + 1) % 10}",
                "fouls_home": fouls_home,
                "fouls_away": fouls_away,
                "fouls_total": fouls_total,
                "xfouls_home": float(rng.uniform(10, 16)),
                "xfouls_away": float(rng.uniform(10, 16)),
                "forma_fouls_home": float(rng.uniform(10, 16)),
                "forma_fouls_away": float(rng.uniform(10, 16)),
                "home_fouls_committed_avg": float(rng.uniform(10, 16)),
                "away_fouls_committed_avg": float(rng.uniform(10, 16)),
                "ref_home_delta": float(rng.uniform(-2, 2)),
                "ref_away_delta": float(rng.uniform(-2, 2)),
                "urgency_home": float(rng.uniform(0.3, 0.8)),
                "urgency_away": float(rng.uniform(0.3, 0.8)),
                "momentum_home": float(rng.uniform(0.3, 0.8)),
                "momentum_away": float(rng.uniform(0.3, 0.8)),
            }
        )
    return matches


def _make_match() -> dict:
    """Single match dict for prediction."""
    return {
        "home_team": "TeamA",
        "away_team": "TeamB",
        "fouls_home": 13,
        "fouls_away": 15,
        "fouls_total": 28,
        "xfouls_home": 12.5,
        "xfouls_away": 13.5,
        "forma_fouls_home": 12.0,
        "forma_fouls_away": 13.0,
        "home_fouls_committed_avg": 13.0,
        "away_fouls_committed_avg": 14.0,
        "ref_home_delta": 0.5,
        "ref_away_delta": -0.5,
        "urgency_home": 0.5,
        "urgency_away": 0.5,
        "momentum_home": 0.5,
        "momentum_away": 0.5,
    }


# ---------------------------------------------------------------------------
# TestRatioLogitModel
# ---------------------------------------------------------------------------

class TestRatioLogitModel:
    def test_parameter_shapes(self) -> None:
        """RatioLogitModel().beta.shape == (6,) and intercept.shape == ()."""
        from src.models.regression import RatioLogitModel

        model = RatioLogitModel()
        assert model.beta.shape == (6,), f"Expected beta shape (6,), got {model.beta.shape}"
        assert model.intercept.shape == (), f"Expected intercept shape (), got {model.intercept.shape}"

    def test_forward_output_shape(self) -> None:
        """forward(torch.zeros(4, 6)) returns shape (4,)."""
        from src.models.regression import RatioLogitModel

        model = RatioLogitModel()
        X = torch.zeros(4, 6)
        out = model(X)
        assert out.shape == (4,), f"Expected output shape (4,), got {out.shape}"


# ---------------------------------------------------------------------------
# TestHomeFoulRatioEstimatorFit
# ---------------------------------------------------------------------------

class TestHomeFoulRatioEstimatorFit:
    def test_fit_sets_is_fitted(self) -> None:
        """After fit(matches), _is_fitted is True."""
        from src.models.regression import HomeFoulRatioEstimator

        estimator = HomeFoulRatioEstimator()
        matches = _make_matches(40)
        estimator.fit(matches)
        assert estimator._is_fitted is True

    def test_fit_stores_normalization(self) -> None:
        """After fit, _feature_means and _feature_stds are ndarrays of length 6."""
        from src.models.regression import HomeFoulRatioEstimator

        estimator = HomeFoulRatioEstimator()
        matches = _make_matches(40)
        estimator.fit(matches)
        assert isinstance(estimator._feature_means, np.ndarray)
        assert isinstance(estimator._feature_stds, np.ndarray)
        assert len(estimator._feature_means) == 6
        assert len(estimator._feature_stds) == 6


# ---------------------------------------------------------------------------
# TestPredictRatio
# ---------------------------------------------------------------------------

class TestPredictRatio:
    def test_returns_float(self) -> None:
        """predict_ratio(match) returns float."""
        from src.models.regression import HomeFoulRatioEstimator

        estimator = HomeFoulRatioEstimator()
        estimator.fit(_make_matches(40))
        result = estimator.predict_ratio(_make_match())
        assert isinstance(result, float)

    def test_within_bounds(self) -> None:
        """predict_ratio result is in [0.30, 0.70]."""
        from src.models.regression import HomeFoulRatioEstimator

        estimator = HomeFoulRatioEstimator()
        estimator.fit(_make_matches(40))
        result = estimator.predict_ratio(_make_match())
        assert 0.30 <= result <= 0.70, f"Expected ratio in [0.30, 0.70], got {result}"

    def test_raises_if_not_fitted(self) -> None:
        """RuntimeError raised if predict_ratio called before fit()."""
        from src.models.regression import HomeFoulRatioEstimator

        estimator = HomeFoulRatioEstimator()
        with pytest.raises(RuntimeError):
            estimator.predict_ratio(_make_match())


# ---------------------------------------------------------------------------
# TestEnsembleRatioCheckpoint
# ---------------------------------------------------------------------------

class TestEnsembleRatioCheckpoint:
    @pytest.mark.xfail(reason="Requires a full trained ensemble — integration-level test")
    def test_save_creates_ratio_checkpoint(self, tmp_path) -> None:
        """After ensemble.save(), ratio_estimator.pt exists in the directory."""
        from src.models.ensemble import FoulPredictionEnsemble

        ensemble = FoulPredictionEnsemble()
        matches = _make_matches(40)
        # This requires full training data — mark as xfail for unit test context
        raise NotImplementedError("Integration test — needs full ensemble data")

    @pytest.mark.xfail(reason="Requires a full trained ensemble — integration-level test")
    def test_load_restores_ratio_estimator(self, tmp_path) -> None:
        """After load(), ratio_estimator._is_fitted is True."""
        from src.models.ensemble import FoulPredictionEnsemble

        ensemble = FoulPredictionEnsemble()
        raise NotImplementedError("Integration test — needs full ensemble data")


# ---------------------------------------------------------------------------
# TestLegacyFallback
# ---------------------------------------------------------------------------

class TestLegacyFallback:
    def test_heuristic_when_no_estimator(self) -> None:
        """With ratio_estimator=None, predict_team_fouls returns dict without error."""
        from src.models.ensemble import FoulPredictionEnsemble
        from src.utils.distributions import FoulPMF
        import numpy as np

        ensemble = FoulPredictionEnsemble()
        # Manually bypass fitted check by calling _heuristic_team_split directly
        # We create a minimal MatchPrediction-like object
        from src.models.ensemble import MatchPrediction

        probs = np.ones(61) / 61
        pmf = FoulPMF(probs=probs)
        total_pred = MatchPrediction(
            match_info=_make_match(),
            pmf_total=pmf,
            pmf_bayes=pmf,
            pmf_regression=pmf,
            pmf_anfis=pmf,
            weights=np.array([1 / 3, 1 / 3, 1 / 3]),
            expected_fouls=28.0,
        )

        # ratio_estimator is None by default
        assert ensemble.ratio_estimator is None

        result = ensemble._heuristic_team_split(_make_match(), total_pred)
        assert isinstance(result, dict)
        assert "home_expected" in result
        assert "away_expected" in result
        assert "total_expected" in result
