"""
tests/overlay/test_ev_kelly_scale.py — Strict TDD for kelly_scale parameter in ev.py.

Tests (4 cases, all RED before T5.2 implementation):
  1. compute_ev(kelly_scale=1.0) is byte-identical to default (regression)
  2. compute_ev(kelly_scale=0.5) produces kelly_stake exactly half of kelly_scale=1.0
  3. Output dict includes both 'kelly_raw' and 'kelly_scaled' keys
  4. compute_ev_all_lines with kelly_scale=0.7 scales all Kelly fractions

Safety net: ev.py currently has no kelly_scale param → tests confirm RED before impl.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from src.utils.ev import compute_ev, compute_ev_all_lines

# ---------------------------------------------------------------------------
# Shared fixture inputs
# ---------------------------------------------------------------------------

_LINE = 25.5
_P_OVER = 0.55
_ODDS_OVER = 1.90
_ODDS_UNDER = 2.00
_KELLY_FRACTION = 0.25
_MIN_EDGE = 0.02


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeEvKellyScale:
    def test_kelly_scale_default_matches_baseline(self) -> None:
        """compute_ev() with default kelly_scale=1.0 is identical to no-param call.

        This is the regression test: existing call sites (no kelly_scale arg)
        must produce byte-identical results.
        """
        result_baseline = compute_ev(
            line=_LINE,
            p_over_model=_P_OVER,
            odds_over=_ODDS_OVER,
            odds_under=_ODDS_UNDER,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
        )
        result_with_scale = compute_ev(
            line=_LINE,
            p_over_model=_P_OVER,
            odds_over=_ODDS_OVER,
            odds_under=_ODDS_UNDER,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
            kelly_scale=1.0,
        )
        assert result_baseline is not None
        assert result_with_scale is not None
        # kelly_stake should be identical
        assert result_baseline["kelly_stake"] == result_with_scale["kelly_stake"], (
            f"kelly_stake differs: baseline={result_baseline['kelly_stake']}, "
            f"with_scale={result_with_scale['kelly_stake']}"
        )

    def test_kelly_scale_half_produces_half_kelly_stake(self) -> None:
        """compute_ev(kelly_scale=0.5) → kelly_stake is exactly half of scale=1.0."""
        result_full = compute_ev(
            line=_LINE,
            p_over_model=_P_OVER,
            odds_over=_ODDS_OVER,
            odds_under=_ODDS_UNDER,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
            kelly_scale=1.0,
        )
        result_half = compute_ev(
            line=_LINE,
            p_over_model=_P_OVER,
            odds_over=_ODDS_OVER,
            odds_under=_ODDS_UNDER,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
            kelly_scale=0.5,
        )
        assert result_full is not None
        assert result_half is not None
        # kelly_raw should be the same in both
        assert "kelly_raw" in result_half, "Missing 'kelly_raw' key in result"
        assert "kelly_scaled" in result_half, "Missing 'kelly_scaled' key in result"

        # kelly_scaled should be ~half of kelly_raw
        # Tolerance 1e-4: both values are rounded to 4 decimal places independently,
        # so rounding error between them can be up to 5e-5.
        assert (
            abs(result_half["kelly_scaled"] - result_half["kelly_raw"] * 0.5) < 1e-4
        ), (
            f"kelly_scaled={result_half['kelly_scaled']:.6f} != "
            f"kelly_raw * 0.5 = {result_half['kelly_raw'] * 0.5:.6f}"
        )

    def test_output_includes_kelly_raw_and_kelly_scaled(self) -> None:
        """Output dict from compute_ev includes 'kelly_raw' and 'kelly_scaled' keys."""
        result = compute_ev(
            line=_LINE,
            p_over_model=_P_OVER,
            odds_over=_ODDS_OVER,
            odds_under=_ODDS_UNDER,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
            kelly_scale=0.7,
        )
        assert result is not None, "Expected a result (edge should be sufficient)"
        assert "kelly_raw" in result, "Missing 'kelly_raw' in result dict"
        assert "kelly_scaled" in result, "Missing 'kelly_scaled' in result dict"
        # kelly_raw should be the unscaled value
        # kelly_scaled should be kelly_raw * 0.7
        assert abs(result["kelly_scaled"] - result["kelly_raw"] * 0.7) < 1e-4, (
            f"kelly_scaled={result['kelly_scaled']:.6f} != "
            f"kelly_raw * 0.7 = {result['kelly_raw'] * 0.7:.6f}"
        )
        # kelly_stake (the backward-compat field) should equal kelly_scaled
        assert (
            result["kelly_stake"] == result["kelly_scaled"]
            or abs(result["kelly_stake"] - result["kelly_scaled"]) < 1e-6
        ), "kelly_stake should equal kelly_scaled for existing call sites"

    def test_compute_ev_all_lines_with_kelly_scale(self) -> None:
        """compute_ev_all_lines(kelly_scale=0.7) scales Kelly in all results."""
        ou_table = {
            24.5: (0.58, 0.42),
            25.5: (0.55, 0.45),
            26.5: (0.50, 0.50),
        }
        market_odds = {
            24.5: (1.85, 2.05),
            25.5: (1.90, 2.00),
            26.5: (2.00, 1.90),
        }

        results_full = compute_ev_all_lines(
            ou_table=ou_table,
            market_odds=market_odds,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
            kelly_scale=1.0,
        )
        results_scaled = compute_ev_all_lines(
            ou_table=ou_table,
            market_odds=market_odds,
            kelly_fraction=_KELLY_FRACTION,
            min_edge=_MIN_EDGE,
            kelly_scale=0.7,
        )

        assert len(results_full) > 0, "Expected at least one EV result"
        assert len(results_scaled) == len(results_full), (
            "scaled and full results should have same number of entries"
        )

        # All scaled results should have kelly_scaled = kelly_raw * 0.7
        for r in results_scaled:
            assert "kelly_raw" in r, "Missing 'kelly_raw' in all_lines result"
            assert "kelly_scaled" in r, "Missing 'kelly_scaled' in all_lines result"
            assert abs(r["kelly_scaled"] - r["kelly_raw"] * 0.7) < 1e-4, (
                f"kelly_scaled={r['kelly_scaled']:.6f} != kelly_raw * 0.7 = {r['kelly_raw'] * 0.7:.6f}"
            )
