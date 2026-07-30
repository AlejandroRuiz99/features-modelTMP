"""
PMF utilities for foul count distributions.
Handles conversion between full probability mass functions,
over/under probabilities, and interval probabilities.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


MAX_K = 61  # support: 0, 1, ..., 60 fouls


@dataclass
class FoulPMF:
    """Discrete probability mass function over foul counts."""

    probs: np.ndarray  # shape (MAX_K,), probs[k] = P(F=k)

    def __post_init__(self):
        self.probs = np.asarray(self.probs, dtype=np.float64)
        if self.probs.shape != (MAX_K,):
            raise ValueError(f"PMF must have shape ({MAX_K},), got {self.probs.shape}")
        total = self.probs.sum()
        if total > 0:
            self.probs /= total

    @property
    def cdf(self) -> np.ndarray:
        return np.cumsum(self.probs)

    @property
    def mean(self) -> float:
        return float(np.dot(np.arange(MAX_K), self.probs))

    @property
    def variance(self) -> float:
        mu = self.mean
        return float(np.dot((np.arange(MAX_K) - mu) ** 2, self.probs))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def entropy(self) -> float:
        p = self.probs[self.probs > 0]
        return float(-np.sum(p * np.log(p)))  # type: ignore[operator]

    def prob_over(self, line: float) -> float:
        """P(F > line). For line=25.5, returns P(F >= 26)."""
        k_min = int(np.floor(line)) + 1
        if k_min >= MAX_K:
            return 0.0
        return float(self.probs[k_min:].sum())

    def prob_under(self, line: float) -> float:
        """P(F < line). For line=25.5, returns P(F <= 25)."""
        return 1.0 - self.prob_over(line)

    def prob_interval(self, low: int, high: int) -> float:
        """P(low <= F <= high)."""
        low = max(0, low)
        high = min(MAX_K - 1, high)
        return float(self.probs[low : high + 1].sum())

    def prob_exactly(self, k: int) -> float:
        if 0 <= k < MAX_K:
            return float(self.probs[k])
        return 0.0

    def over_under_table(
        self, lines: list[float] | None = None
    ) -> dict[float, tuple[float, float]]:
        """Returns {line: (P_over, P_under)} for common betting lines."""
        if lines is None:
            lines = [x + 0.5 for x in range(15, 46)]
        return {line: (self.prob_over(line), self.prob_under(line)) for line in lines}

    def interval_table(self, breakpoints: list[int] | None = None) -> dict[str, float]:
        """Returns {interval_label: probability} for foul bands."""
        if breakpoints is None:
            breakpoints = [0, 20, 25, 30, 35, 40, 60]
        result = {}
        for i in range(len(breakpoints) - 1):
            low, high = breakpoints[i], breakpoints[i + 1]
            if i < len(breakpoints) - 2:
                label = f"{low}-{high - 1}"
                result[label] = self.prob_interval(low, high - 1)
            else:
                label = f"{low}+"
                result[label] = self.prob_interval(low, MAX_K - 1)
        return result

    def mode(self) -> int:
        return int(np.argmax(self.probs))

    def median(self) -> int:
        return int(np.searchsorted(self.cdf, 0.5))

    def quantile(self, q: float) -> int:
        return int(np.searchsorted(self.cdf, q))


def pmf_from_poisson(lam: float) -> FoulPMF:
    """Create PMF from a Poisson distribution with rate lambda."""
    from scipy.stats import poisson

    probs = poisson.pmf(np.arange(MAX_K), mu=lam)
    return FoulPMF(probs=probs)


def pmf_from_negbin(mu: float, alpha: float) -> FoulPMF:
    """
    Create PMF from Negative Binomial (mean-dispersion parameterization).
    mu: mean, alpha: dispersion (variance = mu + alpha * mu^2).
    """
    from scipy.stats import nbinom

    if alpha <= 0:
        return pmf_from_poisson(mu)
    r = 1.0 / alpha
    p = r / (r + mu)
    probs = nbinom.pmf(np.arange(MAX_K), n=r, p=p)
    return FoulPMF(probs=probs)


def pmf_from_intervals(
    interval_probs: np.ndarray,
    breakpoints: list[int],
    smooth: bool = False,
) -> FoulPMF:
    """
    Convert interval probabilities to a full PMF.

    Args:
        interval_probs: Probability mass assigned to each interval.
        breakpoints: Left endpoints of each interval. The final interval extends
            from the last breakpoint to ``MAX_K``.
        smooth: When ``False`` (default), interior intervals use uniform
            distribution and the last uses right-triangular decay — identical
            to the original behaviour. When ``True``, each interval's mass is
            distributed with a triangular (tent) kernel centred at the interval
            midpoint, spanning ±half-width into adjacent intervals. This
            eliminates hard step-function discontinuities at boundaries. The
            resulting PMF is normalised to sum to 1.0.

    Returns:
        A :class:`FoulPMF` whose probabilities sum to 1.0 ± 1e-6.
    """
    if not smooth:
        # ------------------------------------------------------------------ #
        # Original piecewise-uniform implementation (backward compat)
        # ------------------------------------------------------------------ #
        probs = np.zeros(MAX_K)
        n_intervals = len(interval_probs)

        for i, p_interval in enumerate(interval_probs):
            low = breakpoints[i]
            high = breakpoints[i + 1] if i + 1 < len(breakpoints) else MAX_K
            n_vals = high - low
            if n_vals <= 0:
                continue

            is_last = i == n_intervals - 1
            if is_last and n_vals > 1:
                # Right-triangular decay: weight linearly decreasing from low to high
                weights = np.arange(n_vals, 0, -1, dtype=np.float64)
                weights /= weights.sum()
                probs[low:high] = p_interval * weights
            else:
                probs[low:high] = p_interval / n_vals

        return FoulPMF(probs=probs)

    # ---------------------------------------------------------------------- #
    # Triangular blending implementation (smooth=True)
    # ---------------------------------------------------------------------- #
    # Build the right-edge of each interval (exclusive).
    n_intervals = len(interval_probs)
    rights: list[int] = []
    for i in range(n_intervals):
        rights.append(breakpoints[i + 1] if i + 1 < len(breakpoints) else MAX_K)

    # Compute interval half-widths for the tent-function bases.
    #   half_width[i] = (rights[i] - breakpoints[i]) / 2.0
    half_widths: list[float] = [
        (rights[i] - breakpoints[i]) / 2.0 for i in range(n_intervals)
    ]

    k_vals = np.arange(MAX_K, dtype=np.float64)
    probs = np.zeros(MAX_K)

    for i, p_interval in enumerate(interval_probs):
        low = breakpoints[i]
        high = rights[i]
        mid = (low + high) / 2.0
        hw = half_widths[i]

        if hw <= 0.0:
            # Degenerate (zero-width) interval: concentrate at breakpoint.
            idx = int(low)
            if 0 <= idx < MAX_K:
                probs[idx] += p_interval
            continue

        # Tent kernel centred at interval midpoint, with base = 2 × interval width.
        # The kernel spans ±hw BEYOND each interval edge into neighbouring intervals,
        # which smooths the hard steps at breakpoints.
        #   kernel(k) = max(0, 1 - |k - mid| / (2 * hw))
        # where 2*hw = interval_width.  Peak at mid, zero at mid ± 2*hw.
        tent = np.maximum(0.0, 1.0 - np.abs(k_vals - mid) / (2.0 * hw))

        tent_sum = tent.sum()
        if tent_sum > 0.0:
            tent /= tent_sum  # normalise to unit mass

        probs += p_interval * tent

    # Final normalisation guard (should be a no-op if inputs sum to 1).
    total = probs.sum()
    if total > 0.0:
        probs /= total

    return FoulPMF(probs=probs)


def mixture_pmf(pmfs: list[FoulPMF], weights: np.ndarray) -> FoulPMF:
    """Weighted mixture of multiple PMFs."""
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    combined = np.zeros(MAX_K)
    for pmf, w in zip(pmfs, weights):
        combined += w * pmf.probs
    return FoulPMF(probs=combined)


def tilt_pmf_to_mean(
    base_pmf: FoulPMF, target_mean: float, max_iter: int = 80
) -> FoulPMF:
    """
    Recalibra una PMF para que su media se acerque a `target_mean`
    preservando forma relativa (exponential tilting):
        q_k ∝ p_k * exp(lambda * k)

    Se usa bisección sobre lambda; la media resultante es monótona en lambda.
    """
    k = np.arange(MAX_K, dtype=np.float64)
    p = np.asarray(base_pmf.probs, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)

    target = float(np.clip(target_mean, 0.0, MAX_K - 1))

    # Si target ya está muy cerca, no retocar.
    if abs(base_pmf.mean - target) < 1e-6:
        return FoulPMF(probs=p)

    def mean_from_lambda(lmb: float) -> float:
        w = p * np.exp(np.clip(lmb * k, -100.0, 100.0))
        s = w.sum()
        if s <= 1e-18:
            return base_pmf.mean
        q = w / s
        return float(np.dot(k, q))

    lo, hi = -1.5, 1.5
    m_lo, m_hi = mean_from_lambda(lo), mean_from_lambda(hi)
    # Expandimos rango de lambda si hace falta.
    for _ in range(8):
        if target < m_lo:
            hi = lo
            lo *= 2.0
            m_lo = mean_from_lambda(lo)
            continue
        if target > m_hi:
            lo = hi
            hi *= 2.0
            m_hi = mean_from_lambda(hi)
            continue
        break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m_mid = mean_from_lambda(mid)
        if abs(m_mid - target) < 1e-7:
            lo = hi = mid
            break
        if m_mid < target:
            lo = mid
        else:
            hi = mid

    lmb_star = 0.5 * (lo + hi)
    w = p * np.exp(np.clip(lmb_star * k, -100.0, 100.0))
    q = w / max(w.sum(), 1e-12)
    return FoulPMF(probs=q)


def scale_pmf_variance(base_pmf: FoulPMF, variance_scale: float) -> FoulPMF:
    """
    Ajusta la dispersión de una PMF mediante temperature-smoothing:
      q_k ∝ p_k^(1/T), con T = variance_scale.

    - T > 1: PMF más plana (más varianza)
    - T < 1: PMF más picuda (menos varianza)

    Después se reancla la media al valor original para evitar sesgo.
    """
    t = float(max(variance_scale, 1e-3))
    if abs(t - 1.0) < 1e-6:
        return FoulPMF(probs=base_pmf.probs.copy())

    p = np.asarray(base_pmf.probs, dtype=np.float64)
    q = np.power(np.clip(p, 1e-12, 1.0), 1.0 / t)
    q = q / max(q.sum(), 1e-12)
    return tilt_pmf_to_mean(FoulPMF(probs=q), base_pmf.mean)
