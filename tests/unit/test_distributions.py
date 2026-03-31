"""Unit tests for prediction_models/src/utils/distributions.py."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.utils.distributions import (
    MAX_K,
    FoulPMF,
    pmf_from_poisson,
    pmf_from_negbin,
    pmf_from_intervals,
    mixture_pmf,
    tilt_pmf_to_mean,
    scale_pmf_variance,
)


class TestFoulPMFInit:
    """Tests for FoulPMF initialization."""

    def test_valid_pmf_shape(self) -> None:
        """PMF con shape correcto se crea."""
        probs = np.ones(MAX_K) / MAX_K
        pmf = FoulPMF(probs=probs)
        assert pmf.probs.shape == (MAX_K,)
        assert abs(pmf.probs.sum() - 1.0) < 1e-10

    def test_invalid_shape_raises(self) -> None:
        """Shape incorrecto lanza ValueError."""
        with pytest.raises(ValueError):
            FoulPMF(probs=np.ones(10))

    def test_normalizes_to_sum_one(self) -> None:
        """PMF se normaliza automáticamente."""
        probs = np.random.rand(MAX_K)
        pmf = FoulPMF(probs=probs)
        assert abs(pmf.probs.sum() - 1.0) < 1e-10


class TestFoulPMFProperties:
    """Tests for FoulPMF properties."""

    @pytest.fixture
    def uniform_pmf(self) -> FoulPMF:
        """PMF uniforme para tests."""
        return FoulPMF(probs=np.ones(MAX_K) / MAX_K)

    @pytest.fixture
    def peaked_pmf(self) -> FoulPMF:
        """PMF concentrada en k=25."""
        probs = np.zeros(MAX_K)
        probs[25] = 1.0
        return FoulPMF(probs=probs)

    def test_mean_uniform(self, uniform_pmf: FoulPMF) -> None:
        """Media de uniforme es (MAX_K-1)/2."""
        expected = (MAX_K - 1) / 2
        assert abs(uniform_pmf.mean - expected) < 0.01

    def test_mean_peaked(self, peaked_pmf: FoulPMF) -> None:
        """Media de PMF concentrada es el valor del pico."""
        assert peaked_pmf.mean == 25.0

    def test_variance_uniform(self, uniform_pmf: FoulPMF) -> None:
        """Varianza de uniforme es (n²-1)/12."""
        # Varianza de uniforme discreta [0, n-1] = (n²-1)/12
        expected_var = (MAX_K**2 - 1) / 12
        assert abs(uniform_pmf.variance - expected_var) < 0.5

    def test_variance_peaked(self, peaked_pmf: FoulPMF) -> None:
        """Varianza de PMF concentrada es 0."""
        assert abs(peaked_pmf.variance) < 1e-10

    def test_std_is_sqrt_variance(self, uniform_pmf: FoulPMF) -> None:
        """Std es sqrt de variance."""
        assert abs(uniform_pmf.std - math.sqrt(uniform_pmf.variance)) < 1e-10

    def test_entropy_uniform(self, uniform_pmf: FoulPMF) -> None:
        """Entropía de uniforme es ln(MAX_K)."""
        assert abs(uniform_pmf.entropy - math.log(MAX_K)) < 0.01

    def test_cdf_is_cumsum(self, uniform_pmf: FoulPMF) -> None:
        """CDF es cumsum de probs."""
        expected = np.cumsum(uniform_pmf.probs)
        np.testing.assert_array_almost_equal(uniform_pmf.cdf, expected)


class TestFoulPMFMethods:
    """Tests for FoulPMF methods."""

    @pytest.fixture
    def poisson_pmf(self) -> FoulPMF:
        """PMF Poisson con lambda=25."""
        return pmf_from_poisson(25.0)

    def test_prob_over(self, poisson_pmf: FoulPMF) -> None:
        """P(F > line) calcula correctamente."""
        # P(F > 25.5) = P(F >= 26)
        p_over = poisson_pmf.prob_over(25.5)
        p_under = poisson_pmf.prob_under(25.5)
        assert abs(p_over + p_under - 1.0) < 1e-10

    def test_prob_under_line_0(self, poisson_pmf: FoulPMF) -> None:
        """P(F < 0.5) ≈ P(F = 0)."""
        # Small numerical difference due to floating point
        assert poisson_pmf.prob_under(0.5) == pytest.approx(
            poisson_pmf.prob_exactly(0), rel=1e-3
        )

    def test_prob_over_high_line_is_zero(self, poisson_pmf: FoulPMF) -> None:
        """P(F > 60) ≈ 0."""
        assert poisson_pmf.prob_over(60.5) < 1e-10

    def test_prob_interval(self, poisson_pmf: FoulPMF) -> None:
        """P(20 <= F <= 30) = sum de probs en ese rango."""
        p_interval = poisson_pmf.prob_interval(20, 30)
        expected = sum(poisson_pmf.probs[20:31])
        assert abs(p_interval - expected) < 1e-10

    def test_prob_exactly(self, poisson_pmf: FoulPMF) -> None:
        """P(F = k) = probs[k]."""
        for k in [0, 10, 25, 40]:
            assert poisson_pmf.prob_exactly(k) == poisson_pmf.probs[k]

    def test_prob_exactly_out_of_range(self, poisson_pmf: FoulPMF) -> None:
        """P(F = k) = 0 para k fuera de rango."""
        assert poisson_pmf.prob_exactly(-1) == 0.0
        assert poisson_pmf.prob_exactly(100) == 0.0

    def test_mode(self, poisson_pmf: FoulPMF) -> None:
        """Mode es argmax de probs."""
        mode = poisson_pmf.mode()
        assert mode == int(np.argmax(poisson_pmf.probs))

    def test_median(self, poisson_pmf: FoulPMF) -> None:
        """Median es el smallest k donde CDF >= 0.5."""
        median = poisson_pmf.median()
        assert poisson_pmf.cdf[median] >= 0.5

    def test_quantile(self, poisson_pmf: FoulPMF) -> None:
        """Quantile es smallest k donde CDF >= q."""
        q25 = poisson_pmf.quantile(0.25)
        q50 = poisson_pmf.quantile(0.50)
        q75 = poisson_pmf.quantile(0.75)
        assert q25 <= q50 <= q75

    def test_over_under_table(self, poisson_pmf: FoulPMF) -> None:
        """Over/under table tiene formato correcto."""
        table = poisson_pmf.over_under_table(lines=[20.5, 25.5, 30.5])
        assert len(table) == 3
        for line, (p_over, p_under) in table.items():
            assert abs(p_over + p_under - 1.0) < 1e-10

    def test_interval_table(self, poisson_pmf: FoulPMF) -> None:
        """Interval table tiene formato correcto."""
        table = poisson_pmf.interval_table()
        total = sum(table.values())
        assert abs(total - 1.0) < 1e-3  # Suma ≈1(con decimales)


class TestPmfFromPoisson:
    """Tests for pmf_from_poisson."""

    def test_mean_matches_lambda(self) -> None:
        """Media de PMF Poisson ≈ lambda."""
        for lam in [10.0, 25.0, 40.0]:
            pmf = pmf_from_poisson(lam)
            assert abs(pmf.mean - lam) < 0.5

    def test_distribution_normalizes(self) -> None:
        """PMF se normaliza."""
        pmf = pmf_from_poisson(25.0)
        assert abs(pmf.probs.sum() - 1.0) < 1e-10

    def test_mode_approximately_lambda(self) -> None:
        """Mode de Poisson ≈ floor(lambda)."""
        pmf = pmf_from_poisson(25.7)
        assert pmf.mode() in [24, 25, 26]


class TestPmfFromNegbin:
    """Tests for pmf_from_negbin."""

    def test_mean_matches_mu(self) -> None:
        """Media de NegBin ≈ mu."""
        pmf = pmf_from_negbin(mu=25.0, alpha=0.1)
        assert abs(pmf.mean - 25.0) < 0.5

    def test_variance_larger_than_poisson(self) -> None:
        """NegBin tiene varianza > media (overdispersion)."""
        pmf = pmf_from_negbin(mu=25.0, alpha=0.1)
        assert pmf.variance > pmf.mean

    def test_alpha_zero_falls_back_to_poisson(self) -> None:
        """alpha <= 0 devuelve Poisson."""
        pmf_nb = pmf_from_negbin(mu=25.0, alpha=0.0)
        pmf_poi = pmf_from_poisson(25.0)
        np.testing.assert_array_almost_equal(pmf_nb.probs, pmf_poi.probs)


class TestPmfFromIntervals:
    """Tests for pmf_from_intervals."""

    def test_uniform_interior_intervals(self) -> None:
        """Intervalos internos tienen distribución uniforme."""
        # 3 intervalos: [0,10), [10,20), [20,30)+ tail
        probs = np.array([0.3, 0.4, 0.2, 0.1])
        breakpoints = [0, 10, 20, 30]
        pmf = pmf_from_intervals(probs, breakpoints)

        # En [0, 10), cada valor debe tener ~0.03 prob (0.3/10)
        for k in range(10):
            assert abs(pmf.probs[k] - 0.03) < 1e-10

    def test_triangular_tail(self) -> None:
        """Último intervalo usa triangular decay."""
        # Tail en [30, MAX_K) con prob 0.1
        probs = np.array([0.9, 0.1])
        breakpoints = [0, 30]
        pmf = pmf_from_intervals(probs, breakpoints)

        # Primer valor del tail debe tener más prob que el último
        tail_start = 30
        assert pmf.probs[tail_start] > pmf.probs[tail_start + 10]

    def test_sum_is_one(self) -> None:
        """PMF resultante suma 1."""
        probs = np.array([0.2, 0.3, 0.3, 0.2])
        breakpoints = [0, 10, 20, 30, MAX_K]
        pmf = pmf_from_intervals(probs, breakpoints)
        assert abs(pmf.probs.sum() - 1.0) < 1e-10


class TestMixturePmf:
    """Tests for mixture_pmf."""

    def test_mixture_of_two_poisson(self) -> None:
        """Mezcla de dos Poisson."""
        pmf1 = pmf_from_poisson(20.0)
        pmf2 = pmf_from_poisson(30.0)
        mixed = mixture_pmf([pmf1, pmf2], np.array([0.5, 0.5]))

        # Media debe ser ~25 (promedio)
        assert abs(mixed.mean - 25.0) < 1.0

    def test_mixture_weights_normalize(self) -> None:
        """Weights se normalizan automáticamente."""
        pmf1 = pmf_from_poisson(20.0)
        pmf2 = pmf_from_poisson(30.0)
        mixed = mixture_pmf([pmf1, pmf2], np.array([3.0, 1.0]))  # 75/25

        assert abs(mixed.mean - 22.5) < 1.0  # Closer to first component

    def test_mixture_preserves_sum(self) -> None:
        """PMF mezclada suma 1."""
        pmf1 = pmf_from_poisson(20.0)
        pmf2 = pmf_from_poisson(30.0)
        mixed = mixture_pmf([pmf1, pmf2], np.array([0.7, 0.3]))
        assert abs(mixed.probs.sum() - 1.0) < 1e-10


class TestTiltPmfToMean:
    """Tests for tilt_pmf_to_mean."""

    def test_increases_mean(self) -> None:
        """Tilt aumenta la media."""
        pmf = pmf_from_poisson(20.0)
        tilted = tilt_pmf_to_mean(pmf, target_mean=25.0)
        assert tilted.mean > pmf.mean
        assert abs(tilted.mean - 25.0) < 0.1

    def test_decreases_mean(self) -> None:
        """Tilt disminuye la media."""
        pmf = pmf_from_poisson(30.0)
        tilted = tilt_pmf_to_mean(pmf, target_mean=25.0)
        assert tilted.mean < pmf.mean
        assert abs(tilted.mean - 25.0) < 0.1

    def test_same_mean_no_change(self) -> None:
        """Si target ≈ media actual, no cambia."""
        pmf = pmf_from_poisson(25.0)
        tilted = tilt_pmf_to_mean(pmf, target_mean=25.0)
        np.testing.assert_array_almost_equal(tilted.probs, pmf.probs)

    def test_preserves_sum(self) -> None:
        """PMF tilteda suma 1."""
        pmf = pmf_from_poisson(20.0)
        tilted = tilt_pmf_to_mean(pmf, target_mean=30.0)
        assert abs(tilted.probs.sum() - 1.0) < 1e-10


class TestScalePmfVariance:
    """Tests for scale_pmf_variance."""

    def test_increase_variance(self) -> None:
        """T > 1 aumenta varianza."""
        pmf = pmf_from_poisson(25.0)
        original_var = pmf.variance
        scaled = scale_pmf_variance(pmf, variance_scale=1.5)
        assert scaled.variance > original_var

    def test_decrease_variance(self) -> None:
        """T < 1 disminuye varianza."""
        pmf = pmf_from_negbin(mu=25.0, alpha=0.3)  # Overdispersed
        original_var = pmf.variance
        scaled = scale_pmf_variance(pmf, variance_scale=0.5)
        assert scaled.variance < original_var

    def test_preserves_mean(self) -> None:
        """Preserva la media (se re-ancla)."""
        pmf = pmf_from_poisson(25.0)
        original_mean = pmf.mean
        scaled = scale_pmf_variance(pmf, variance_scale=1.5)
        assert abs(scaled.mean - original_mean) < 0.1

    def test_variance_scale_one_no_change(self) -> None:
        """T = 1 no cambia la PMF."""
        pmf = pmf_from_poisson(25.0)
        scaled = scale_pmf_variance(pmf, variance_scale=1.0)
        np.testing.assert_array_almost_equal(scaled.probs, pmf.probs)
