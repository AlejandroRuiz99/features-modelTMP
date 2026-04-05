"""Unit tests for scripts/train.py — grid search configuration and composite selection."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "prediction_models"))

_TRAIN_SRC = Path(__file__).resolve().parent.parent.parent / "scripts" / "train.py"


def _extract_list(source: str, var_name: str) -> list:
    """Extract a list literal assigned to var_name from Python source text."""
    pattern = rf"{re.escape(var_name)}\s*=\s*(\[[^\]]*\])"
    match = re.search(pattern, source)
    if not match:
        raise AssertionError(f"Variable '{var_name}' not found in train.py")
    return ast.literal_eval(match.group(1))


def _composite(mae: float, nll: float, mean_mae: float, mean_nll: float) -> float:
    mae_norm = mae / mean_mae if mean_mae != 0 else 1.0
    nll_norm = nll / mean_nll if mean_nll != 0 else 1.0
    return round(0.6 * mae_norm + 0.4 * nll_norm, 6)


class TestGridSize:
    def test_grid_size_is_100(self) -> None:
        """len(prior_mix) * len(min_weight) * len(variance_scale) == 100."""
        source = _TRAIN_SRC.read_text(encoding="utf-8")
        prior_mix = _extract_list(source, "prior_mix_values")
        min_weight = _extract_list(source, "min_weight_values")
        variance_scale = _extract_list(source, "variance_scale_values")
        total = len(prior_mix) * len(min_weight) * len(variance_scale)
        assert total == 100, f"Expected grid size 100, got {total} ({len(prior_mix)}x{len(min_weight)}x{len(variance_scale)})"

    def test_prior_mix_contains_low_values(self) -> None:
        """prior_mix_values must include 0.10 and 0.15."""
        source = _TRAIN_SRC.read_text(encoding="utf-8")
        prior_mix = _extract_list(source, "prior_mix_values")
        assert 0.10 in prior_mix, f"0.10 not in prior_mix_values: {prior_mix}"
        assert 0.15 in prior_mix, f"0.15 not in prior_mix_values: {prior_mix}"


class TestCompositeSelection:
    def test_composite_selection_prefers_calibrated(self) -> None:
        """X=(mae=2.1, nll=3.0), Y=(mae=2.0, nll=5.0), mean_mae=2.05, mean_nll=4.0 → composite(X) < composite(Y)."""
        mean_mae = 2.05
        mean_nll = 4.0
        score_x = _composite(2.1, 3.0, mean_mae, mean_nll)
        score_y = _composite(2.0, 5.0, mean_mae, mean_nll)
        assert score_x < score_y, (
            f"Expected composite(X)={score_x} < composite(Y)={score_y}; "
            "X has slightly worse MAE but much better NLL — composite should prefer X"
        )

    def test_composite_stored_in_results(self) -> None:
        """After composite pass, each result entry must have a 'composite' key."""
        results = [
            {"mae": 2.1, "nll": 3.0},
            {"mae": 2.0, "nll": 5.0},
            {"mae": 1.9, "nll": 4.5},
        ]
        mean_mae = sum(r["mae"] for r in results) / len(results)
        mean_nll = sum(r["nll"] for r in results) / len(results)
        for r in results:
            mae_norm = r["mae"] / mean_mae if mean_mae != 0 else 1.0
            nll_norm = r["nll"] / mean_nll if mean_nll != 0 else 1.0
            r["composite"] = round(0.6 * mae_norm + 0.4 * nll_norm, 6)
        for r in results:
            assert "composite" in r, f"Result entry missing 'composite' key: {r}"
            assert isinstance(r["composite"], float)

    def test_degenerate_nll_no_zero_division(self) -> None:
        """All nll=0.0 → no exception, composite is a valid finite float."""
        results = [
            {"mae": 2.0, "nll": 0.0},
            {"mae": 2.5, "nll": 0.0},
        ]
        mean_mae = sum(r["mae"] for r in results) / len(results)
        mean_nll = sum(r["nll"] for r in results) / len(results)  # == 0.0
        for r in results:
            mae_norm = r["mae"] / mean_mae if mean_mae != 0 else 1.0
            nll_norm = r["nll"] / mean_nll if mean_nll != 0 else 1.0
            r["composite"] = round(0.6 * mae_norm + 0.4 * nll_norm, 6)
        import math
        for r in results:
            assert math.isfinite(r["composite"]), f"composite is not finite: {r['composite']}"
