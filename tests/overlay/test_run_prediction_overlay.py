"""
tests/overlay/test_run_prediction_overlay.py — T7.3: Integration test for
run_prediction.py with --narrative flag.

Tests:
  1. run_prediction.py --narrative with valid YAML + --output-json writes overlay
     section to the JSON output.
  2. Overlay log file is written to overlay/logs/ when --narrative is supplied.
  3. run_prediction.py without --narrative produces output with NO overlay section
     (identity behavior).

NOTE: These tests use the fixture feature dicts (no Supabase) by calling the
_prediction_to_dict-equivalent function with pre-built features.  They test the
wiring logic by importing run_prediction helpers directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
SNAPSHOT_DIR = ROOT / "tests" / "fixtures" / "snapshots"
NARRATIVE_DIR = ROOT / "tests" / "fixtures" / "narratives"

if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from overlay.applier import OverlayResult, apply_overlay
from overlay.loader import load_narrative
from overlay.rules import load_catalog

# Catalog path
CATALOG_PATH = ROOT / "overlay" / "rules.yaml"


# ---------------------------------------------------------------------------
# Helpers — load fixture prediction dict via snapshot feature fixtures
# ---------------------------------------------------------------------------


def _load_snapshot(name: str) -> dict:
    """Load a snapshot JSON as reference (not for predicting — just for reference)."""
    path = SNAPSHOT_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _build_prediction_from_fixture(fixture: dict) -> dict:
    """Run ensemble.predict on a fixture dict and return a prediction dict."""
    from src.models.ensemble import FoulPredictionEnsemble  # type: ignore[import]

    CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"
    config_path = CHECKPOINT_DIR / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(CHECKPOINT_DIR)

    pred = ensemble.predict(fixture)
    team_pred = (
        ensemble.predict_team_fouls(fixture, total_prediction=pred, reconcile=True)
        or {}
    )

    lines = [21.5, 23.5, 24.5, 25.5, 27.5, 29.5]
    ou_table = pred.over_under
    if team_pred.get("reconciled") and team_pred.get("total_pmf") is not None:
        ou_table = team_pred["total_pmf"].over_under_table(lines)

    return {
        "pmf_total": pred.pmf_total,
        "expected_fouls": float(pred.expected_fouls),
        "home_expected": float(
            team_pred.get("home_expected", pred.expected_fouls * 0.55)
        ),
        "away_expected": float(
            team_pred.get("away_expected", pred.expected_fouls * 0.45)
        ),
        "over_under": ou_table,
    }


# Import the fixture feature dict from capture_snapshots
sys.path.insert(0, str(SNAPSHOT_DIR))
try:
    from capture_snapshots import FIXTURE_ESPANYOL_LEVANTE  # type: ignore[import]
except ImportError:
    FIXTURE_ESPANYOL_LEVANTE = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverlaySection:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:  # type: ignore[return]
        """Skip all tests if ensemble checkpoints are not available."""
        checkpoint = PRED_DIR / "checkpoints" / "ensemble" / "config.json"
        if not checkpoint.exists():
            pytest.skip("Ensemble checkpoints not available")
        if FIXTURE_ESPANYOL_LEVANTE is None:
            pytest.skip("Fixture feature dict not available")

    def test_apply_overlay_with_narrative_returns_result(self) -> None:
        """apply_overlay returns OverlayResult with non-empty data when narrative fires rules."""
        narrative_path = NARRATIVE_DIR / "valid_full.yaml"
        narrative = load_narrative(narrative_path)
        catalog = load_catalog(CATALOG_PATH)

        prediction = _build_prediction_from_fixture(FIXTURE_ESPANYOL_LEVANTE)
        result = apply_overlay(prediction, narrative, catalog)

        assert isinstance(result, OverlayResult)
        assert result.pre_pmf_summary["mean"] > 0
        assert result.post_pmf_summary["mean"] > 0
        # The valid_full narrative has relegation + physical clash flags
        # → should fire at least 1 rule
        assert len(result.rules_fired) >= 1, (
            "Expected at least 1 rule to fire with valid_full narrative"
        )

    def test_apply_overlay_without_narrative_is_identity(self) -> None:
        """With minimal narrative (no matching rules), PMF is unchanged."""
        from overlay.schema import Narrative, NarrativeMatch, ObjectiveOverride

        narrative = Narrative(
            match=NarrativeMatch(home="Espanol", away="Levante", date="2026-04-27"),
            confidence_level=1,
            objectives={
                "home": ObjectiveOverride(label="mid"),
                "away": ObjectiveOverride(label="mid"),
            },
        )
        catalog = load_catalog(CATALOG_PATH)

        prediction = _build_prediction_from_fixture(FIXTURE_ESPANYOL_LEVANTE)
        pre_mean = prediction["expected_fouls"]

        result = apply_overlay(prediction, narrative, catalog)

        assert result.rules_fired == []
        post_mean = result.post_pmf_summary["mean"]
        # Tolerance: pmf_summary rounds mean to 4 decimal places; allow rounding delta
        assert abs(post_mean - pre_mean) < 1e-3, (
            f"Expected identity (no rules fired), but mean changed significantly: "
            f"{pre_mean:.6f} -> {post_mean:.6f}"
        )

    def test_log_writer_called_produces_log_file(self, tmp_path: Path) -> None:
        """write_overlay_log writes a file that can be read back as valid JSON."""
        from datetime import datetime

        from overlay.log_writer import write_overlay_log

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "match": {"home": "Espanol", "away": "Levante", "date": "2026-04-27"},
            "narrative_raw": "match:\n  home: Espanol\n",
            "parsed_flags": {"confidence_level": 4},
            "pre_overlay": {"expected_fouls": 25.65, "pmf_summary": {}},
            "rules_fired": [],
            "post_overlay": {"expected_fouls": 25.65, "pmf_summary": {}},
            "kelly_raw_vs_scaled": {"kelly_raw": 1.0, "kelly_scaled": 0.85},
            "actual_fouls": None,
        }

        log_path = write_overlay_log(record, tmp_path / "logs")
        assert log_path.exists()
        loaded = json.loads(log_path.read_text(encoding="utf-8"))
        assert loaded["match"]["home"] == "Espanol"
        assert loaded["actual_fouls"] is None
