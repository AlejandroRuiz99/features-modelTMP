"""Failing tests for pmf_from_intervals triangular blending (T10).

These tests document the expected behaviour of the ``smooth`` parameter
that will be added to ``pmf_from_intervals`` in T11.

All tests that reference ``smooth=True`` will FAIL until T11 is implemented
because the parameter does not yet exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "prediction_models")
)

import numpy as np
import pytest

from src.utils.distributions import MAX_K, FoulPMF, pmf_from_intervals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_probs(n: int) -> np.ndarray:
    """Return equal-weight interval probabilities."""
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# T10-1: Backward compatibility — smooth=False must match current output
# ---------------------------------------------------------------------------


class TestPmfFromIntervalsBackwardCompat:
    """smooth=False (the default) must produce identical output to the
    current piecewise-uniform implementation."""

    def test_smooth_false_identical_to_no_arg(self) -> None:
        """pmf_from_intervals(probs, bps, smooth=False) == pmf_from_intervals(probs, bps)."""
        probs = np.array([0.2, 0.3, 0.3, 0.2])
        breakpoints = [0, 10, 20, 30]

        pmf_default = pmf_from_intervals(probs, breakpoints)
        pmf_explicit_false = pmf_from_intervals(probs, breakpoints, smooth=False)

        np.testing.assert_array_equal(
            pmf_default.probs,
            pmf_explicit_false.probs,
            err_msg="smooth=False must produce the same result as calling without smooth arg",
        )

    def test_smooth_false_interior_uniform(self) -> None:
        """Interior intervals are piecewise-uniform when smooth=False."""
        probs = np.array([0.4, 0.6])
        breakpoints = [0, 10, 20]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=False)

        # First interval [0, 10) — each of the 10 bins must be 0.4/10
        expected_first = 0.4 / 10
        for k in range(10):
            assert abs(pmf.probs[k] - expected_first) < 1e-10, (
                f"smooth=False: probs[{k}] should be {expected_first}, got {pmf.probs[k]}"
            )

    def test_smooth_false_sum_is_one(self) -> None:
        """PMF with smooth=False sums to 1.0."""
        probs = np.array([0.15, 0.35, 0.35, 0.15])
        breakpoints = [0, 15, 25, 35]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=False)

        assert abs(pmf.probs.sum() - 1.0) < 1e-6, (
            f"smooth=False PMF should sum to 1.0, got {pmf.probs.sum()}"
        )


# ---------------------------------------------------------------------------
# T10-2: Smoothed output differs from piecewise-uniform
# ---------------------------------------------------------------------------


class TestPmfFromIntervalsSmoothed:
    """smooth=True must produce output that differs from smooth=False."""

    def test_smooth_true_differs_from_smooth_false(self) -> None:
        """smooth=True output must differ from smooth=False output."""
        probs = np.array([0.2, 0.3, 0.3, 0.2])
        breakpoints = [0, 10, 20, 30]

        pmf_hard = pmf_from_intervals(probs, breakpoints, smooth=False)
        pmf_soft = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert not np.allclose(pmf_hard.probs, pmf_soft.probs), (
            "smooth=True must produce a different PMF from smooth=False"
        )

    def test_smooth_true_returns_foul_pmf(self) -> None:
        """smooth=True must return a FoulPMF instance."""
        probs = np.array([0.25, 0.50, 0.25])
        breakpoints = [0, 15, 30]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert isinstance(pmf, FoulPMF), (
            f"pmf_from_intervals must return FoulPMF, got {type(pmf)}"
        )


# ---------------------------------------------------------------------------
# T10-3: Smooth PMF sums to 1.0 ± 1e-6
# ---------------------------------------------------------------------------


class TestPmfFromIntervalsSmoothNormalization:
    """The smooth PMF must be a valid probability distribution."""

    def test_smooth_sum_to_one_balanced(self) -> None:
        """Balanced 4-interval smooth PMF sums to 1.0."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        breakpoints = [0, 10, 20, 30]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert abs(pmf.probs.sum() - 1.0) < 1e-6, (
            f"Smooth PMF must sum to 1.0 ± 1e-6, got {pmf.probs.sum()}"
        )

    def test_smooth_sum_to_one_skewed(self) -> None:
        """Skewed interval weights still produce a normalized smooth PMF."""
        probs = np.array([0.05, 0.15, 0.50, 0.30])
        breakpoints = [0, 10, 20, 30]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert abs(pmf.probs.sum() - 1.0) < 1e-6, (
            f"Skewed smooth PMF must sum to 1.0 ± 1e-6, got {pmf.probs.sum()}"
        )

    def test_smooth_sum_to_one_8_intervals(self) -> None:
        """8-interval (target config) smooth PMF sums to 1.0."""
        probs = _uniform_probs(8)
        # Mimicking the target percentile breakpoints [12,25,38,50,62,75,88]
        breakpoints = [0, 12, 25, 38, 50, 62, 75, 88]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert abs(pmf.probs.sum() - 1.0) < 1e-6, (
            f"8-interval smooth PMF must sum to 1.0 ± 1e-6, got {pmf.probs.sum()}"
        )

    def test_smooth_probs_all_non_negative(self) -> None:
        """All probabilities in smooth PMF must be ≥ 0."""
        probs = np.array([0.1, 0.4, 0.4, 0.1])
        breakpoints = [0, 10, 20, 30]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert np.all(pmf.probs >= 0), "Smooth PMF must have no negative probabilities"


# ---------------------------------------------------------------------------
# T10-4: No hard discontinuities at breakpoints
# ---------------------------------------------------------------------------


class TestPmfFromIntervalsNoContinuityBreaks:
    """Smooth PMF must not have hard step-function discontinuities at
    interval boundaries.  We measure this by comparing the ratio of adjacent
    values at each breakpoint: a piecewise-uniform PMF has a hard jump
    whenever two adjacent intervals carry different probabilities.
    """

    # The threshold for maximum allowed jump ratio between adjacent bins.
    # Piecewise-uniform produces jumps of arbitrary magnitude; the smooth
    # version should stay within this factor.
    MAX_ADJACENT_RATIO = 3.0

    def _max_boundary_jump(self, pmf: FoulPMF, breakpoints: list[int]) -> float:
        """Return the largest ratio |p[k] / p[k-1]| at any breakpoint k."""
        ratios = []
        for bp in breakpoints[1:-1]:  # interior breakpoints only
            p_left = pmf.probs[bp - 1]
            p_right = pmf.probs[bp]
            if p_left > 1e-12 and p_right > 1e-12:
                ratios.append(max(p_left / p_right, p_right / p_left))
        return float(max(ratios)) if ratios else 1.0

    def test_smooth_reduces_boundary_jump_vs_hard(self) -> None:
        """Smooth PMF must have smaller boundary jumps than piecewise-uniform."""
        # Use very unequal probabilities to force large hard jumps
        probs = np.array([0.05, 0.60, 0.25, 0.10])
        breakpoints = [0, 10, 20, 30]

        pmf_hard = pmf_from_intervals(probs, breakpoints, smooth=False)
        pmf_soft = pmf_from_intervals(probs, breakpoints, smooth=True)

        jump_hard = self._max_boundary_jump(pmf_hard, breakpoints)
        jump_soft = self._max_boundary_jump(pmf_soft, breakpoints)

        assert jump_soft < jump_hard, (
            f"Smooth PMF (max jump ratio {jump_soft:.2f}) must have smaller "
            f"boundary jump than piecewise-uniform ({jump_hard:.2f})"
        )

    def test_smooth_adjacent_values_within_threshold(self) -> None:
        """Adjacent values in smooth PMF differ by at most MAX_ADJACENT_RATIO."""
        probs = np.array([0.10, 0.40, 0.40, 0.10])
        breakpoints = [0, 10, 20, 30]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        for k in range(1, MAX_K):
            p_prev = pmf.probs[k - 1]
            p_curr = pmf.probs[k]
            if p_prev > 1e-12 and p_curr > 1e-12:
                ratio = max(p_prev / p_curr, p_curr / p_prev)
                assert ratio <= self.MAX_ADJACENT_RATIO, (
                    f"Adjacent ratio at k={k} is {ratio:.2f}, "
                    f"exceeds threshold {self.MAX_ADJACENT_RATIO}"
                )


# ---------------------------------------------------------------------------
# T10-5: Edge cases
# ---------------------------------------------------------------------------


class TestPmfFromIntervalsEdgeCases:
    """Edge cases: single interval, extreme breakpoints."""

    def test_single_interval_smooth_false(self) -> None:
        """Single interval with smooth=False produces valid PMF."""
        probs = np.array([1.0])
        breakpoints = [0]  # one interval: [0, MAX_K)

        pmf = pmf_from_intervals(probs, breakpoints, smooth=False)

        assert isinstance(pmf, FoulPMF)
        assert abs(pmf.probs.sum() - 1.0) < 1e-6

    def test_single_interval_smooth_true(self) -> None:
        """Single interval with smooth=True produces valid PMF summing to 1.0."""
        probs = np.array([1.0])
        breakpoints = [0]  # one interval: [0, MAX_K)

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert isinstance(pmf, FoulPMF)
        assert abs(pmf.probs.sum() - 1.0) < 1e-6

    def test_extreme_breakpoints_smooth_false(self) -> None:
        """Breakpoints at 0 and MAX_K boundary — smooth=False."""
        probs = np.array([0.5, 0.5])
        breakpoints = [0, MAX_K - 1]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=False)

        assert abs(pmf.probs.sum() - 1.0) < 1e-6

    def test_extreme_breakpoints_smooth_true(self) -> None:
        """Breakpoints at 0 and MAX_K boundary — smooth=True sums to 1.0."""
        probs = np.array([0.5, 0.5])
        breakpoints = [0, MAX_K - 1]

        pmf = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert abs(pmf.probs.sum() - 1.0) < 1e-6

    def test_two_intervals_smooth_entropy_higher(self) -> None:
        """Smooth PMF has higher entropy than piecewise-uniform for same probs.

        This captures the 'smoother' property: more spread means higher entropy.
        """
        probs = np.array([0.8, 0.2])
        breakpoints = [0, 20]

        pmf_hard = pmf_from_intervals(probs, breakpoints, smooth=False)
        pmf_soft = pmf_from_intervals(probs, breakpoints, smooth=True)

        assert pmf_soft.entropy > pmf_hard.entropy, (
            f"Smooth PMF entropy ({pmf_soft.entropy:.4f}) must be higher than "
            f"piecewise-uniform entropy ({pmf_hard.entropy:.4f})"
        )
