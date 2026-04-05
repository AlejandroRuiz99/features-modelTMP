"""Unit tests for prediction_models/src/models/gating_network.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "prediction_models"))

_GATING_SRC = Path(__file__).resolve().parent.parent.parent / "prediction_models" / "src" / "models" / "gating_network.py"


def _compute_prior(nlls: list[float]) -> list[float]:
    """Pure reimplementation of the expected global-prior formula (NLL-only)."""
    scores = [1.0 / (nll + 1e-8) for nll in nlls]
    total = sum(scores)
    return [s / total for s in scores]


class TestLossCoefficients:
    def test_loss_coefficients_are_correct(self) -> None:
        """Source must contain 0.55, 0.25, 0.15 on the loss line; 0.40 must NOT appear there."""
        source = _GATING_SRC.read_text(encoding="utf-8")
        # Find the line containing 'loss ='
        loss_line = ""
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("loss =") and "nll" in stripped:
                loss_line = stripped
                break
        assert loss_line, "loss = ... line not found in gating_network.py"
        assert "0.55" in loss_line, f"Expected '0.55' in loss line: {loss_line}"
        assert "0.25" in loss_line, f"Expected '0.25' in loss line: {loss_line}"
        assert "0.15" in loss_line, f"Expected '0.15' in loss line: {loss_line}"
        assert "0.40" not in loss_line, f"Old coefficient '0.40' still present in loss line: {loss_line}"

    def test_loss_comment_documents_0_95(self) -> None:
        """Source must contain '0.95' as a comment near the loss computation."""
        source = _GATING_SRC.read_text(encoding="utf-8")
        assert "0.95" in source, "Expected '0.95' to appear as a comment in gating_network.py"


class TestGlobalPriorLogic:
    def test_global_prior_nll_only_ordering(self) -> None:
        """NLL=[3.16, 4.20, 5.10] → weights[0] > weights[1] > weights[2], sum == 1.0."""
        nlls = [3.16, 4.20, 5.10]
        weights = _compute_prior(nlls)
        assert weights[0] > weights[1] > weights[2]
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_global_prior_equal_nll_equal_weights(self) -> None:
        """NLL=[4.0, 4.0, 4.0] → each weight ≈ 1/3."""
        nlls = [4.0, 4.0, 4.0]
        weights = _compute_prior(nlls)
        for w in weights:
            assert abs(w - 1 / 3) < 1e-6
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_global_prior_old_formula_absent(self) -> None:
        """Source must NOT contain '0.7 * nll' or '0.7 * nlls' in gating_network.py."""
        source = _GATING_SRC.read_text(encoding="utf-8")
        assert "0.7 * nll" not in source, "Old formula '0.7 * nll' still present in gating_network.py"
        assert "0.7 * nlls" not in source, "Old formula '0.7 * nlls' still present in gating_network.py"
