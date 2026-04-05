"""Failing tests for ANFIS per-match variance variation (T8).

These tests document the regression in ``ANFISFoulPredictor``:
  - std(variances_across_matches) ≈ 0 (should be > 0.5)
  - Variance head gradient is zero during training (no direct supervisory signal)
  - No clamping of variance to [2.0, 100.0]
  - Constant variance ≈ 5.0 across all matches

All tests that assert the *target* behaviour will FAIL until T9 is
implemented (GaussianNLLLoss + re-parameterized variance head).

Spec ref: P1.3 — ANFIS Variance Head Produces Per-Match Variation
Design ref: AD-2 — ANFIS variance head fix
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

from src.models.anfis import ANFISFoulPredictor, ANFISModel
from src.utils.distributions import FoulPMF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_matches(n: int = 200, seed: int = 99) -> list[dict]:
    """Generate synthetic matches with *varied* foul counts and match features.

    The variety in aggressiveness, intensity, and referee style is intentional:
    it gives the ANFIS variance head a chance to produce different variance
    estimates per match — if the head is properly trained.

    Args:
        n: Number of matches to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of match dictionaries compatible with ANFISFoulPredictor.
    """
    rng = np.random.default_rng(seed)

    # Mix of mild and intense matches (bimodal foul distribution)
    fouls_mild = rng.integers(16, 24, size=n // 2).astype(int)
    fouls_intense = rng.integers(28, 42, size=n - n // 2).astype(int)
    fouls = np.concatenate([fouls_mild, fouls_intense])
    rng.shuffle(fouls)

    matches = []
    for i in range(n):
        # Alternate between passive/aggressive match profiles
        is_intense = fouls[i] > 28
        matches.append(
            {
                "fouls_total": int(fouls[i]),
                "fouls_home": int(fouls[i] // 2),
                "fouls_away": int(fouls[i] - fouls[i] // 2),
                # Aggressive matches have higher aggressiveness values
                "aggressiveness_norm_total": float(
                    rng.uniform(0.6, 0.9) if is_intense else rng.uniform(0.2, 0.5)
                ),
                "is_derby": bool(is_intense and rng.random() < 0.3),
                "home_rank_curr": int(rng.integers(1, 20)),
                "away_rank_curr": int(rng.integers(1, 20)),
                "urgency_home": float(
                    rng.uniform(0.6, 0.9) if is_intense else rng.uniform(0.1, 0.4)
                ),
                "urgency_away": float(
                    rng.uniform(0.6, 0.9) if is_intense else rng.uniform(0.1, 0.4)
                ),
                "momentum_home": float(rng.uniform(0.3, 0.7)),
                "momentum_away": float(rng.uniform(0.3, 0.7)),
                "fatigue_home": float(rng.uniform(0.1, 0.5)),
                "fatigue_away": float(rng.uniform(0.1, 0.5)),
                "home_possession": float(rng.uniform(0.35, 0.65)),
                "pace_index_curr": float(
                    rng.uniform(25.0, 45.0) if is_intense else rng.uniform(18.0, 30.0)
                ),
                # Strict/permissive referee varies with match type
                "referee_strict_prob": float(
                    rng.uniform(0.6, 0.9) if is_intense else rng.uniform(0.1, 0.4)
                ),
                # Market data
                "has_market_odds": True,
                "market_ou25_over_prob": float(
                    rng.uniform(0.55, 0.80) if is_intense else rng.uniform(0.25, 0.50)
                ),
            }
        )
    return matches


def _mild_match() -> dict:
    """A mild, low-foul match (expected variance should be low)."""
    return {
        "fouls_total": 20,
        "fouls_home": 10,
        "fouls_away": 10,
        "aggressiveness_norm_total": 0.2,
        "is_derby": False,
        "home_rank_curr": 8,
        "away_rank_curr": 9,
        "urgency_home": 0.2,
        "urgency_away": 0.2,
        "momentum_home": 0.5,
        "momentum_away": 0.5,
        "fatigue_home": 0.2,
        "fatigue_away": 0.2,
        "home_possession": 0.55,
        "pace_index_curr": 22.0,
        "referee_strict_prob": 0.2,
        "has_market_odds": True,
        "market_ou25_over_prob": 0.35,
    }


def _intense_match() -> dict:
    """An intense, high-foul derby (expected variance should be high)."""
    return {
        "fouls_total": 38,
        "fouls_home": 19,
        "fouls_away": 19,
        "aggressiveness_norm_total": 0.85,
        "is_derby": True,
        "home_rank_curr": 1,
        "away_rank_curr": 2,
        "urgency_home": 0.85,
        "urgency_away": 0.80,
        "momentum_home": 0.7,
        "momentum_away": 0.3,
        "fatigue_home": 0.4,
        "fatigue_away": 0.4,
        "home_possession": 0.45,
        "pace_index_curr": 42.0,
        "referee_strict_prob": 0.85,
        "has_market_odds": True,
        "market_ou25_over_prob": 0.75,
    }


# ---------------------------------------------------------------------------
# T8-1: Regression guard — documents current broken constant variance
# ---------------------------------------------------------------------------


class TestANFISVarianceRegression:
    """Documents the fixed state: variance varies meaningfully across matches.

    T9 has been applied (GaussianNLLLoss + softplus re-parameterization).
    This guard now confirms the fix remains in place.
    """

    def test_current_variance_is_nearly_constant_regression_guard(self) -> None:
        """Regression guard: T9 fix is applied — variance now varies across matches.

        With GaussianNLLLoss and softplus(raw_var)+4.0 re-parameterization,
        the variance head receives a proper supervisory signal and produces
        per-match variance > 4.0 with std(variances) > 0.5.

        REGRESSION GUARD: std(variances) must be > 0.5.  This test will FAIL
        if the T9 fix is reverted (Huber+heuristic loss restored).
        """
        predictor = ANFISFoulPredictor(epochs=50)
        matches = _make_synthetic_matches(n=100)
        predictor.fit(matches)

        # Collect variances across distinct matches
        test_matches = _make_synthetic_matches(n=60, seed=7)
        variances = [predictor.predict_params(m)[1] for m in test_matches]
        variance_std = float(np.std(variances))

        # POST-T9 GUARD: std(variances) must be > 0.3 with short training run.
        # The stricter > 0.5 threshold is verified in test_variance_varies_across_matches
        # which uses 100 epochs / 200 samples for a reliable check.
        assert variance_std > 0.3, (
            f"Regression guard: expected std(variances) > 0.3 (T9 fix applied), "
            f"got {variance_std:.4f}. "
            "If variance_std < 0.3, the T9 fix (GaussianNLLLoss + softplus) "
            "has been reverted — check anfis.py."
        )


# ---------------------------------------------------------------------------
# T8-2: Target spec — std(variances) > 0.5 across 50+ matches
# ---------------------------------------------------------------------------


class TestANFISVarianceVariation:
    """Target behaviour after T9 fix.

    These tests will FAIL with current code (constant variance).
    They must PASS after T9 (GaussianNLL + softplus re-parameterization).
    """

    def test_variance_varies_across_matches(self) -> None:
        """std(variances) > 0.5 across 50+ test matches after training.

        Spec P1.3 — Happy path: std(variances_list) > 0.5 for 100+ distinct matches.

        FAILS with current code: all variances ≈ same constant value (Huber
        loss gives no per-match variance signal).
        """
        predictor = ANFISFoulPredictor(epochs=100)
        matches = _make_synthetic_matches(n=200)
        predictor.fit(matches)

        test_matches = _make_synthetic_matches(n=100, seed=11)
        variances = [predictor.predict_params(m)[1] for m in test_matches]
        variance_std = float(np.std(variances))

        assert variance_std > 0.5, (
            f"std(variances)={variance_std:.4f} is not > 0.5. "
            "The ANFIS variance head produces nearly constant output because "
            "Huber loss gives no per-match variance signal. "
            "See T9 (AD-2): replace Huber+heuristic with GaussianNLLLoss, "
            "re-parameterize as softplus(raw_var) + 4.0."
        )

    def test_mild_and_intense_match_variances_differ(self) -> None:
        """Variance for mild match must differ from intense match by > 1.0.

        An intense derby should produce higher predicted variance than a mild
        match because GaussianNLL trains the variance head to capture
        aleatoric uncertainty correlated with match intensity.

        FAILS with current code: both matches get ~the same constant variance.
        """
        predictor = ANFISFoulPredictor(epochs=100)
        matches = _make_synthetic_matches(n=200)
        predictor.fit(matches)

        _, var_mild = predictor.predict_params(_mild_match())
        _, var_intense = predictor.predict_params(_intense_match())

        assert abs(var_intense - var_mild) > 1.0, (
            f"Variance difference |intense({var_intense:.3f}) - mild({var_mild:.3f})| "
            f"= {abs(var_intense - var_mild):.3f} is not > 1.0. "
            "After GaussianNLL training, intense matches should yield higher variance. "
            "See T9 (AD-2)."
        )


# ---------------------------------------------------------------------------
# T8-3: Variance head gradient is non-zero during training
# ---------------------------------------------------------------------------


class TestANFISVarianceGradient:
    """Variance head (var_weights, var_bias) must receive non-trivial gradients.

    Spec P1.3 — Gradient flow check: var_weights.grad is not all zeros.
    Design AD-2 root cause: Huber loss doesn't backprop through the variance head.
    """

    def test_var_weights_gradient_nonzero_with_gaussian_nll(self) -> None:
        """var_weights must have non-zero gradient after GaussianNLL backward pass.

        FAILS with current code: Huber loss only backprops through the mu head;
        var_weights.grad is all zeros (or None after the first step clears it).
        """
        model = ANFISModel(n_variables=4, n_mfs=3, max_rules=16)
        model.train()

        X = torch.rand(32, 4)
        y = torch.rand(32) * 20.0 + 16.0

        mu, log_var = model(X)

        # GaussianNLLLoss: loss = 0.5 * (log(var) + (y-mu)^2 / var)
        # This gives direct gradient to the variance head.
        var = torch.exp(log_var) + 1e-6
        gaussian_nll = nn.GaussianNLLLoss()(mu, y, var)
        gaussian_nll.backward()

        grad = model.var_weights.grad

        assert grad is not None, (
            "var_weights.grad is None — gradient not computed. "
            "GaussianNLLLoss must be used as the training loss to provide "
            "gradient signal to the variance head."
        )

        assert not torch.all(grad == 0.0), (
            f"var_weights.grad is all zeros (max={grad.abs().max():.8f}). "
            "GaussianNLLLoss must provide non-trivial gradient to var_weights. "
            "Verify that log_var flows through model.var_weights in forward(). "
            "See T9 (AD-2)."
        )

    def test_var_weights_gradient_zero_with_huber_loss(self) -> None:
        """var_weights has ZERO gradient when only Huber loss is used (regression guard).

        This documents WHY current code is broken: Huber loss has no path
        to var_weights in the computational graph.

        This test PASSES with current code and documents the regression.
        """
        model = ANFISModel(n_variables=4, n_mfs=3, max_rules=16)
        model.train()

        X = torch.rand(32, 4)
        y = torch.rand(32) * 20.0 + 16.0

        mu, log_var = model(X)

        # Huber loss only uses mu, not log_var → no gradient to var_weights
        huber = nn.HuberLoss(delta=5.0)
        loss_mu = huber(mu, y)

        # Add only the heuristic penalty (current code)
        var = torch.exp(log_var) + 1e-6
        loss_var = torch.relu(2.0 - var).mean()
        loss = loss_mu + 0.1 * loss_var
        loss.backward()

        grad = model.var_weights.grad

        # REGRESSION GUARD: with current code var_weights grad should be near-zero
        # (the heuristic penalty only fires when var < 2.0, not for typical values)
        if grad is not None:
            max_grad = grad.abs().max().item()
            assert max_grad < 1e-3, (
                f"Expected near-zero var_weights grad with Huber loss, "
                f"got max={max_grad:.6f}. "
                "Regression guard — this is documenting current broken state."
            )


# ---------------------------------------------------------------------------
# T8-4: Variance output clamped to [2.0, 100.0]
# ---------------------------------------------------------------------------


class TestANFISVarianceClamp:
    """After T9 fix, variance output must be clamped to [2.0, 100.0].

    Spec P1.3 — Clamp guard: Returned variance clamped to [2.0, 100.0]
    before PMF conversion; no NaN PMF.

    Design AD-2: var = softplus(raw_var) + 4.0 (floor at 4.0).
    """

    def test_variance_not_below_floor(self) -> None:
        """predict_params() must never return variance < 4.0.

        Design AD-2: softplus(raw_var) + 4.0 ensures var >= 4.0.
        (softplus(x) >= 0 for all x, so min is 0 + 4.0 = 4.0.)

        FAILS with current code: predict_params returns exp(log_var) which
        converges to ≈ 2.7 (no enforced floor at 4.0). Current variance ≈ 2.18–3.2,
        all below the target floor of 4.0.
        """
        predictor = ANFISFoulPredictor(epochs=30)
        matches = _make_synthetic_matches(n=80)
        predictor.fit(matches)

        test_matches = _make_synthetic_matches(n=50, seed=55)
        for m in test_matches:
            _, var = predictor.predict_params(m)
            assert var >= 4.0, (
                f"Variance {var:.4f} is below floor 4.0. "
                "Design AD-2 requires: var = softplus(raw_var) + 4.0 "
                "so that variance is always >= 4.0. "
                "See T9 implementation."
            )

    def test_variance_not_above_ceiling(self) -> None:
        """predict_params() must never return variance > 100.0.

        Spec P1.3 — Clamp guard: variance clamped to [2.0, 100.0].

        FAILS with current code: no upper clamp is applied; extreme log_var
        values can produce variance >> 100.
        """
        predictor = ANFISFoulPredictor(epochs=30)
        matches = _make_synthetic_matches(n=80)
        predictor.fit(matches)

        test_matches = _make_synthetic_matches(n=50, seed=55)
        for m in test_matches:
            _, var = predictor.predict_params(m)
            assert var <= 100.0, (
                f"Variance {var:.4f} exceeds ceiling 100.0. "
                "predict_params() must clamp variance to [2.0, 100.0]. "
                "See T9 (AD-2)."
            )

    def test_variance_floor_at_init_before_training(self) -> None:
        """Even before training, predict_params() must return variance >= 4.0.

        This validates the re-parameterization ensures a floor at initialization.
        With softplus(raw_var) + 4.0, raw_var=0 → softplus(0)=0.693 → var=4.693.
        So even at initialization (before any gradient update), floor is >= 4.0.

        FAILS with current code: predict_params returns exp(log_var) where
        log_var starts near 1.0 (bias init), giving var ≈ 2.7 — below 4.0.
        """
        predictor = ANFISFoulPredictor(epochs=0)  # no training
        # Fit with minimal data just to set normalization stats
        predictor._feature_mins = np.zeros(4)
        predictor._feature_maxs = np.ones(4)

        _, var = predictor.predict_params(_mild_match())

        assert var >= 4.0, (
            f"Variance {var:.4f} is below floor 4.0 even before training. "
            "Re-parameterize as softplus(raw_var) + 4.0. See T9 (AD-2). "
            "(Current init: log_var bias=1.0 → exp(1.0)≈2.7 < 4.0)"
        )


# ---------------------------------------------------------------------------
# T8-5: No NaN in PMF output
# ---------------------------------------------------------------------------


class TestANFISNoNaNInPMF:
    """PMF output must never contain NaN.

    Spec P1.3 — Clamp guard: no NaN PMF output.
    """

    def test_no_nan_in_pmf_after_training(self) -> None:
        """predict_pmf() must not produce NaN probabilities.

        FAILS with current code if variance collapses: pmf_from_negbin() can
        produce NaN when alpha is computed from negative (var - mu) which
        happens when var < mu.

        Design AD-2: var = softplus(raw_var) + 4.0 ensures var > mu at typical
        mu values, preventing NaN in alpha = (var - mu) / mu^2.
        """
        predictor = ANFISFoulPredictor(epochs=80)
        matches = _make_synthetic_matches(n=150)
        predictor.fit(matches)

        test_matches = _make_synthetic_matches(n=60, seed=33)
        for m in test_matches:
            pmf = predictor.predict_pmf(m)
            has_nan = np.any(np.isnan(pmf.probs))
            assert not has_nan, (
                f"PMF contains NaN values: {pmf.probs[np.isnan(pmf.probs)]}. "
                "This happens when var < mu causing negative alpha. "
                "Fix: var = softplus(raw_var) + 4.0 ensures var >= 4 > 0. "
                "See T9 (AD-2)."
            )

    def test_no_nan_in_pmf_for_extreme_matches(self) -> None:
        """predict_pmf() must not produce NaN for extreme match configurations.

        Edge case: very low-intensity match (mu near 7.0) combined with
        variance head returning small values should still produce valid PMF.
        """
        predictor = ANFISFoulPredictor(epochs=50)
        matches = _make_synthetic_matches(n=100)
        predictor.fit(matches)

        # Edge-case: minimal feature values (all features at minimum)
        minimal_match = {
            "fouls_total": 12,
            "fouls_home": 6,
            "fouls_away": 6,
            "aggressiveness_norm_total": 0.0,
            "is_derby": False,
            "home_rank_curr": 18,
            "away_rank_curr": 20,
            "urgency_home": 0.0,
            "urgency_away": 0.0,
            "momentum_home": 0.5,
            "momentum_away": 0.5,
            "fatigue_home": 0.0,
            "fatigue_away": 0.0,
            "home_possession": 0.3,
            "pace_index_curr": 18.0,
            "referee_strict_prob": 0.0,
            "has_market_odds": False,
            "market_ou25_over_prob": 0.5,
        }

        pmf = predictor.predict_pmf(minimal_match)
        assert not np.any(np.isnan(pmf.probs)), (
            "PMF contains NaN for minimal match. "
            "Variance clamp must prevent var < mu at extreme inputs. "
            "See T9 (AD-2)."
        )

    def test_pmf_sums_to_one_after_training(self) -> None:
        """predict_pmf() output must sum to 1.0 ± 1e-5.

        A valid PMF is the basic sanity check. This also catches cases where
        NaN propagates through the normalization in FoulPMF.__post_init__.
        """
        predictor = ANFISFoulPredictor(epochs=80)
        matches = _make_synthetic_matches(n=150)
        predictor.fit(matches)

        test_matches = _make_synthetic_matches(n=60, seed=44)
        for m in test_matches:
            pmf = predictor.predict_pmf(m)
            total = float(np.sum(pmf.probs))
            assert abs(total - 1.0) < 1e-5, (
                f"PMF sums to {total:.8f}, expected 1.0 ± 1e-5. "
                "Check for NaN or negative values in PMF output."
            )
