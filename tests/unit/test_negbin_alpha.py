"""Failing tests for NegBin alpha regularization (T6).

These tests document the regression in ``NegBinRegressor``:
  - alpha ≈ 0.25–0.30 after training (should be < 0.10)
  - PMF std ≈ 12.4 after training (should be in [4.5, 7.5])
  - No alpha penalty in the loss function

All tests that assert the *target* behaviour will FAIL until T7 is
implemented (clamped _log_alpha + alpha penalty in neg_log_likelihood).

Spec ref: P1.2 — NegBin Alpha Regularization
Design ref: AD-1 — NegBin alpha regularization strategy
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "prediction_models")
)

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.regression import FoulRegressionPredictor, NegBinRegressor
from src.utils.distributions import FoulPMF, pmf_from_negbin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_matches(n: int = 320, seed: int = 42) -> list[dict]:
    """Generate synthetic match dicts with realistic La Liga foul statistics.

    Fouls total drawn from NegBin(mu=26, alpha=0.03) to simulate well-behaved
    La Liga data.  Feature values are sampled from plausible ranges so that
    _extract_features() produces valid vectors without needing real data.

    Args:
        n: Number of matches to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of match dictionaries compatible with FoulRegressionPredictor.
    """
    rng = np.random.default_rng(seed)

    # Low-alpha NegBin: alpha=0.03 => tight dist, std ≈ 5.7 at mu=26
    mu_true = 26.0
    alpha_true = 0.03
    r = 1.0 / alpha_true
    p = r / (r + mu_true)
    fouls = np.round(
        np.random.default_rng(seed).negative_binomial(r, p, size=n)
    ).astype(int)
    fouls = np.clip(fouls, 12, 55)

    matches = []
    for i in range(n):
        matches.append(
            {
                "fouls_total": int(fouls[i]),
                "fouls_home": int(fouls[i] // 2),
                "fouls_away": int(fouls[i] - fouls[i] // 2),
                "home_fouls_committed_avg": float(rng.uniform(9.0, 15.0)),
                "home_fouls_suffered_avg": float(rng.uniform(9.0, 15.0)),
                "away_fouls_committed_avg": float(rng.uniform(9.0, 15.0)),
                "away_fouls_suffered_avg": float(rng.uniform(9.0, 15.0)),
                "home_fouls_committed_curr": float(rng.uniform(9.0, 15.0)),
                "away_fouls_committed_curr": float(rng.uniform(9.0, 15.0)),
                "rank_diff_norm": float(rng.uniform(-1.0, 1.0)),
                "is_derby": bool(rng.random() < 0.1),
                "home_possession": float(rng.uniform(0.35, 0.65)),
                "xg_diff": float(rng.uniform(-1.5, 1.5)),
                # ref features
                "ref_home_delta": float(rng.uniform(-2.0, 2.0)),
                "ref_away_delta": float(rng.uniform(-2.0, 2.0)),
                "ref_pair_delta_sum": float(rng.uniform(-3.0, 3.0)),
                "ref_pair_samples": float(rng.integers(5, 30)),
                "referee_expected_fouls": float(rng.uniform(22.0, 30.0)),
                # market / extended features
                "has_market_odds": True,
                "market_home_win_prob": float(rng.uniform(0.25, 0.55)),
                "market_draw_prob": float(rng.uniform(0.20, 0.30)),
                "market_away_win_prob": float(rng.uniform(0.15, 0.45)),
                "market_ou25_over_prob": float(rng.uniform(0.35, 0.65)),
                "market_favorite_prob": float(rng.uniform(0.40, 0.65)),
                "market_balance": float(rng.uniform(0.8, 1.2)),
                "pace_index_curr": float(rng.uniform(25.0, 42.0)),
                "home_shots_curr": float(rng.uniform(8.0, 15.0)),
                "away_shots_curr": float(rng.uniform(7.0, 13.0)),
                "home_corners_curr": float(rng.uniform(3.0, 7.0)),
                "away_corners_curr": float(rng.uniform(2.5, 6.5)),
                # context features
                "xfouls_home": float(rng.uniform(10.0, 15.0)),
                "xfouls_away": float(rng.uniform(10.0, 15.0)),
                "fouls_provoked_home": float(rng.uniform(9.0, 14.0)),
                "fouls_provoked_away": float(rng.uniform(9.0, 14.0)),
                "forma_fouls_home": float(rng.uniform(10.0, 14.0)),
                "forma_fouls_away": float(rng.uniform(10.0, 14.0)),
                "urgency_home": float(rng.uniform(0.2, 0.8)),
                "urgency_away": float(rng.uniform(0.2, 0.8)),
                "momentum_home": float(rng.uniform(0.3, 0.7)),
                "momentum_away": float(rng.uniform(0.3, 0.7)),
                "days_rest_home": float(rng.uniform(3.0, 14.0)),
                "days_rest_away": float(rng.uniform(3.0, 14.0)),
                "xfouls_factor_home": float(rng.uniform(0.9, 1.1)),
                "xfouls_factor_away": float(rng.uniform(0.9, 1.1)),
            }
        )
    return matches


def _typical_match() -> dict:
    """Return a typical La Liga match dict for prediction checks."""
    return {
        "fouls_total": 26,
        "fouls_home": 13,
        "fouls_away": 13,
        "home_fouls_committed_avg": 12.5,
        "home_fouls_suffered_avg": 12.5,
        "away_fouls_committed_avg": 12.5,
        "away_fouls_suffered_avg": 12.5,
        "home_fouls_committed_curr": 12.5,
        "away_fouls_committed_curr": 12.5,
        "rank_diff_norm": 0.0,
        "is_derby": False,
        "home_possession": 0.5,
        "xg_diff": 0.0,
        "ref_home_delta": 0.0,
        "ref_away_delta": 0.0,
        "ref_pair_delta_sum": 0.0,
        "ref_pair_samples": 20.0,
        "referee_expected_fouls": 26.0,
        "has_market_odds": True,
        "market_home_win_prob": 0.40,
        "market_draw_prob": 0.25,
        "market_away_win_prob": 0.35,
        "market_ou25_over_prob": 0.50,
        "market_favorite_prob": 0.45,
        "market_balance": 1.0,
        "pace_index_curr": 30.0,
        "home_shots_curr": 11.0,
        "away_shots_curr": 10.0,
        "home_corners_curr": 4.5,
        "away_corners_curr": 4.0,
        "xfouls_home": 12.5,
        "xfouls_away": 12.5,
        "fouls_provoked_home": 12.0,
        "fouls_provoked_away": 12.0,
        "forma_fouls_home": 12.5,
        "forma_fouls_away": 12.5,
        "urgency_home": 0.5,
        "urgency_away": 0.5,
        "momentum_home": 0.5,
        "momentum_away": 0.5,
        "days_rest_home": 7.0,
        "days_rest_away": 7.0,
        "xfouls_factor_home": 1.0,
        "xfouls_factor_away": 1.0,
    }


# ---------------------------------------------------------------------------
# T6-1: Regression guard — documents current broken alpha value
# ---------------------------------------------------------------------------


class TestNegBinAlphaRegression:
    """Documents the current (broken) state: alpha is too high after training.

    These tests PASS against current code to confirm the regression exists.
    They are kept as documentation and will continue to pass after the fix
    because they assert the *old* behaviour is NOT present anymore.
    """

    def test_current_alpha_too_high_regression_guard(self) -> None:
        """Regression guard: current code produces alpha >> 0.10.

        This test DOCUMENTS the regression.  It asserts that without the fix,
        alpha converges to a value substantially above 0.10 (typically ~0.25).

        When T7 is implemented (alpha penalty), alpha should drop below 0.10
        and this test will need to be updated or removed.
        """
        predictor = FoulRegressionPredictor(epochs=100)
        matches = _make_synthetic_matches(n=320)
        predictor.fit(matches)

        _, alpha = predictor.predict_params(_typical_match())

        # REGRESSION GUARD: current code gives alpha > 0.10
        # After fix, alpha should be < 0.10
        assert alpha > 0.10, (
            f"Regression guard: expected alpha > 0.10 (broken code), got {alpha:.4f}. "
            "If alpha < 0.10, T7 fix has been applied — update this guard."
        )

    def test_current_pmf_std_too_high_regression_guard(self) -> None:
        """Regression guard: current code produces PMF std >> 7.5.

        Without regularization, alpha ≈ 0.25 gives std ≈ 12.4 at mu=26.
        """
        predictor = FoulRegressionPredictor(epochs=100)
        matches = _make_synthetic_matches(n=320)
        predictor.fit(matches)

        pmf = predictor.predict_pmf(_typical_match())

        # REGRESSION GUARD: current code gives std >> 7.5
        # After fix, std should be in [4.5, 7.5]
        assert pmf.std > 7.5, (
            f"Regression guard: expected PMF std > 7.5 (broken code), got {pmf.std:.4f}. "
            "If std <= 7.5, T7 fix has been applied — update this guard."
        )


# ---------------------------------------------------------------------------
# T6-2: Target spec — alpha < 0.10 after fitting 300+ matches
# ---------------------------------------------------------------------------


class TestNegBinAlphaTarget:
    """Target behaviour after T7 fix.

    These tests will FAIL with current code (alpha not penalized).
    They must PASS after T7 (clamp + penalty) is implemented.
    """

    def test_alpha_below_target_after_fit(self) -> None:
        """After fitting on 300+ matches, alpha must be < 0.10.

        Spec P1.2 — Happy path: alpha < 0.10 for typical La Liga foul data.

        FAILS with current code: alpha ≈ 0.25–0.30 (no penalty in loss).
        """
        predictor = FoulRegressionPredictor(epochs=300)
        matches = _make_synthetic_matches(n=320)
        predictor.fit(matches)

        _, alpha = predictor.predict_params(_typical_match())

        assert alpha < 0.10, (
            f"alpha={alpha:.4f} is not < 0.10. "
            "NegBinRegressor.neg_log_likelihood() must add an alpha penalty "
            "to guide _log_alpha toward low-dispersion. "
            "See T7 (AD-1): add 0.5 * relu(alpha - 0.10)^2 to NLL."
        )

    def test_pmf_std_in_target_range_after_fit(self) -> None:
        """After fitting 300+ matches, PMF std must be in [4.5, 7.5].

        Spec P1.2 — Happy path: predict_pmf(match).std in [4.5, 7.5].

        FAILS with current code: std ≈ 12.4 (alpha too large).
        """
        predictor = FoulRegressionPredictor(epochs=300)
        matches = _make_synthetic_matches(n=320)
        predictor.fit(matches)

        pmf = predictor.predict_pmf(_typical_match())

        assert 4.5 <= pmf.std <= 7.5, (
            f"PMF std={pmf.std:.4f} is not in [4.5, 7.5]. "
            "Expected tight NegBin distribution after alpha regularization. "
            "See T7 (AD-1): clamp _log_alpha to (-6, -1.2) and add penalty to NLL."
        )

    def test_alpha_does_not_exceed_max_clamp(self) -> None:
        """Alpha must never exceed 0.30 (the clamp upper bound from AD-1).

        Design AD-1: _log_alpha clamped to (-6, -1.2) → alpha in [0.002, 0.30].

        FAILS with current code: no clamp is applied in the alpha property.
        """
        predictor = FoulRegressionPredictor(epochs=300)
        matches = _make_synthetic_matches(n=320)
        predictor.fit(matches)

        _, alpha = predictor.predict_params(_typical_match())

        assert alpha <= 0.30, (
            f"alpha={alpha:.4f} exceeds the hard ceiling of 0.30. "
            "Design AD-1 requires _log_alpha clamped to max=-1.2 "
            "(exp(-1.2) ≈ 0.30). See T7 implementation."
        )

    def test_alpha_above_minimum_floor(self) -> None:
        """Alpha must be above 0.001 (numerical floor, not collapsed to 0).

        Design AD-1: alpha in [0.002, 0.30] after clamping _log_alpha to (-6, -1.2).

        This validates the lower bound of the alpha range.
        """
        predictor = FoulRegressionPredictor(epochs=300)
        matches = _make_synthetic_matches(n=320)
        predictor.fit(matches)

        _, alpha = predictor.predict_params(_typical_match())

        assert alpha >= 0.001, (
            f"alpha={alpha:.4f} is below numerical floor 0.001. "
            "This should not happen after clamping _log_alpha."
        )


# ---------------------------------------------------------------------------
# T6-3: Loss function includes alpha penalty term
# ---------------------------------------------------------------------------


class TestNegBinAlphaPenaltyInLoss:
    """The loss function must penalize large alpha values.

    Spec P1.2 — Penalty prevents explosion: penalty term pulls _log_alpha down.
    Design AD-1: add 0.5 * relu(alpha - 0.10)^2 to neg_log_likelihood.

    These tests verify the penalty term exists in the loss by checking that the
    loss EXCEEDS the pure NLL by the expected penalty contribution.
    """

    def test_loss_exceeds_pure_nll_when_alpha_above_threshold(self) -> None:
        """neg_log_likelihood() must exceed pure NB NLL by >= penalty term.

        When alpha=0.30, penalty = 0.5 * (0.30 - 0.10)^2 = 0.02.
        So: loss >= pure_NLL + 0.02.

        With current code: loss == pure_NLL (no penalty), so
        loss - pure_NLL == 0 < 0.02 → test FAILS.

        With T7 fix: loss = NLL + 0.5 * relu(alpha-0.10)^2
        → loss - NLL >= 0.02 → test PASSES.

        FAILS with current code: neg_log_likelihood has no penalty term.
        """
        import math

        n_features = 1
        # alpha = 0.30 → log_alpha = ln(0.30 - 1e-6) ≈ -1.2040
        alpha_test = 0.30
        log_alpha_test = math.log(alpha_test - 1e-6)

        # Expected penalty contribution at alpha=0.30 > 0.10:
        # penalty = 0.5 * relu(0.30 - 0.10)^2 = 0.5 * 0.04 = 0.02
        expected_penalty = 0.5 * max(alpha_test - 0.10, 0.0) ** 2  # == 0.02

        # Build a model fixed at alpha=0.30, intercept=ln(26) (no beta)
        model = NegBinRegressor(n_features=n_features, regularization=0.0)
        with torch.no_grad():
            model._log_alpha.fill_(log_alpha_test)
            model.intercept.fill_(math.log(26.0))
            model.beta.fill_(0.0)

        # Compute pure NB NLL manually (without any penalty)
        X = torch.ones(50, n_features)
        y = torch.full((50,), 26.0)

        with torch.no_grad():
            mu = torch.exp(model.forward(X))
            alpha = model.alpha
            r = 1.0 / alpha
            p = r / (r + mu)
            pure_nll = -(
                torch.lgamma(y + r)
                - torch.lgamma(r)
                - torch.lgamma(y + 1)
                + r * torch.log(p)
                + y * torch.log(1.0 - p + 1e-10)
            ).mean()
            pure_nll_val = float(pure_nll.item())

        # Compute actual loss from neg_log_likelihood()
        actual_loss = model.neg_log_likelihood(X, y).item()

        excess = actual_loss - pure_nll_val

        assert excess >= expected_penalty, (
            f"neg_log_likelihood() excess over pure NLL = {excess:.6f} "
            f"but expected >= {expected_penalty:.6f} (0.5*(alpha-0.10)^2 penalty). "
            "Design AD-1 requires: add `0.5 * torch.relu(alpha - 0.10)**2` "
            "to neg_log_likelihood(). See T7 implementation."
        )

    def test_penalty_is_zero_when_alpha_at_threshold(self) -> None:
        """When alpha == 0.10, the penalty term must contribute exactly 0.

        relu(0.10 - 0.10)^2 = 0, so loss == pure_NLL at the boundary.

        This test PASSES with current code (no penalty at alpha=0.10 is trivially
        met because there is no penalty at all). It is a behavioral specification
        for T7: below the threshold, training is not penalized.
        """
        import math

        n_features = 1
        alpha_threshold = 0.10
        log_alpha_threshold = math.log(alpha_threshold - 1e-6)

        model = NegBinRegressor(n_features=n_features, regularization=0.0)
        with torch.no_grad():
            model._log_alpha.fill_(log_alpha_threshold)
            model.intercept.fill_(math.log(26.0))
            model.beta.fill_(0.0)

        X = torch.ones(50, n_features)
        y = torch.full((50,), 26.0)

        with torch.no_grad():
            mu = torch.exp(model.forward(X))
            alpha = model.alpha
            r = 1.0 / alpha
            p = r / (r + mu)
            pure_nll_val = float(
                -(
                    torch.lgamma(y + r)
                    - torch.lgamma(r)
                    - torch.lgamma(y + 1)
                    + r * torch.log(p)
                    + y * torch.log(1.0 - p + 1e-10)
                )
                .mean()
                .item()
            )

        actual_loss = model.neg_log_likelihood(X, y).item()

        # At alpha=0.10, penalty=0 → |loss - NLL| < 1e-5
        excess = abs(actual_loss - pure_nll_val)
        assert excess < 1e-4, (
            f"|loss - pure_NLL| = {excess:.8f} at alpha≈0.10. "
            "Expected ~0 because penalty = 0.5*relu(0)^2 = 0."
        )


# ---------------------------------------------------------------------------
# T6-4: Alpha clamping in property (no gradient, hard ceiling)
# ---------------------------------------------------------------------------


class TestNegBinAlphaClamp:
    """The alpha property must clamp _log_alpha to a valid range.

    Design AD-1: In NegBinRegressor.alpha property, clamp _log_alpha to
    (-6, -1.2) → alpha range [0.002, 0.30].

    Currently: alpha = exp(_log_alpha) + 1e-6 — no clamping applied.
    """

    def test_alpha_property_clamped_below_max(self) -> None:
        """Setting _log_alpha > -1.2 must still return alpha ≈ exp(-1.2).

        FAILS with current code: no clamp, so alpha = exp(0) + 1e-6 ≈ 1.0.
        """
        model = NegBinRegressor(n_features=2)
        with torch.no_grad():
            model._log_alpha.fill_(2.0)  # unrestricted: alpha ≈ 7.4

        alpha_val = model.alpha.item()

        # With clamp at max=-1.2: exp(-1.2) + 1e-6 ≈ 0.3012
        assert alpha_val <= 0.32, (
            f"alpha={alpha_val:.4f} exceeds max=0.30. "
            "NegBinRegressor.alpha property must clamp _log_alpha: "
            "clamped = torch.clamp(self._log_alpha, min=-6.0, max=-1.2). "
            "See T7 (AD-1)."
        )

    def test_alpha_property_clamped_above_min(self) -> None:
        """Setting _log_alpha < -6 must still return alpha ≈ exp(-6).

        FAILS with current code: no clamp, so alpha = exp(-10) + 1e-6 ≈ 1e-4 ≈ 0.
        """
        model = NegBinRegressor(n_features=2)
        with torch.no_grad():
            model._log_alpha.fill_(-10.0)  # unrestricted: alpha ≈ 4.5e-5

        alpha_val = model.alpha.item()

        # With clamp at min=-6: exp(-6) + 1e-6 ≈ 0.00248
        assert alpha_val >= 0.002, (
            f"alpha={alpha_val:.6f} is below floor=0.002. "
            "NegBinRegressor.alpha property must clamp _log_alpha: "
            "clamped = torch.clamp(self._log_alpha, min=-6.0, max=-1.2). "
            "See T7 (AD-1)."
        )

    def test_alpha_property_returns_tensor(self) -> None:
        """alpha property must return a torch.Tensor (not float).

        This is required for penalty: relu(alpha - 0.10)^2 must be differentiable.
        """
        model = NegBinRegressor(n_features=2)
        alpha = model.alpha

        assert isinstance(alpha, torch.Tensor), (
            f"alpha property must return torch.Tensor, got {type(alpha)}"
        )
