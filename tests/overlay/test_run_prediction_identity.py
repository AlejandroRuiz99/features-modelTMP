"""
tests/overlay/test_run_prediction_identity.py — T7.4: CRITICAL identity regression.

Verifies that running ensemble.predict() on the 3 fixture feature dicts
produces byte-identical output to the pre-overlay Phase 0 snapshots.

These tests prove that the overlay system (when not activated by a narrative)
does NOT change any prediction output values.

Normalization rules:
  - Timestamps are NOT in prediction output (timestamps go ONLY to overlay logs).
  - All numeric comparisons use exact equality (float serialization is deterministic).
  - String/bool fields use exact equality.

Failure protocol (per AGENTS.md):
  - If any of these tests fail, STOP. Do NOT lower the bar.
  - Report status: blocked with the failing field and values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
SNAPSHOT_DIR = ROOT / "tests" / "fixtures" / "snapshots"

if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

# Add snapshot dir so we can import fixture dicts
sys.path.insert(0, str(SNAPSHOT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_snapshot(name: str) -> dict:
    path = SNAPSHOT_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ensemble():  # type: ignore[return]
    from src.models.ensemble import FoulPredictionEnsemble  # type: ignore[import]

    CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"
    config_path = CHECKPOINT_DIR / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(CHECKPOINT_DIR)
    return ensemble


def _prediction_to_dict(ensemble, feat: dict) -> dict:  # type: ignore[return]
    """Same logic as run_prediction._prediction_to_dict — no overlay wiring."""
    pred = ensemble.predict(feat)
    team_pred = (
        ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True) or {}
    )
    return {
        "match": f"{feat['home_team']} vs {feat['away_team']}",
        "date": feat.get("date") or "",
        "jornada": feat.get("matchday") or 0,
        "referee": feat.get("referee") or "",
        "expected_fouls": float(pred.expected_fouls),
        "referee_strict_prob": float(pred.referee_strict_prob),
        "weights": [float(w) for w in pred.weights],
        "home_expected": float(team_pred.get("home_expected", 0.0)),
        "away_expected": float(team_pred.get("away_expected", 0.0)),
        "total_expected": float(team_pred.get("total_expected", pred.expected_fouls)),
        "reconciled": bool(team_pred.get("reconciled", False)),
        "over_under": {
            str(k): {"over": float(v[0]), "under": float(v[1])}
            for k, v in (pred.over_under or {}).items()
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ensemble():  # type: ignore[return]
    """Load ensemble once for all regression tests."""
    checkpoint = PRED_DIR / "checkpoints" / "ensemble" / "config.json"
    if not checkpoint.exists():
        pytest.skip("Ensemble checkpoints not available")

    return _load_ensemble()


@pytest.fixture(scope="module")
def fixture_dicts() -> dict:
    """Import fixture dicts from capture_snapshots.py."""
    try:
        from capture_snapshots import (  # type: ignore[import]
            FIXTURE_ESPANYOL_LEVANTE,
            FIXTURE_MADRID_ATLETICO,
            FIXTURE_VILLARREAL_OSASUNA,
        )

        return {
            "espanyol_levante_j32": FIXTURE_ESPANYOL_LEVANTE,
            "realmadrid_atletico_j29": FIXTURE_MADRID_ATLETICO,
            "villarreal_osasuna_j25": FIXTURE_VILLARREAL_OSASUNA,
        }
    except ImportError:
        pytest.skip("Fixture feature dicts not available")
        return {}  # never reached


class TestIdentityRegression:
    """CRITICAL: These tests must ALL pass. Do not lower bar on failures."""

    def test_espanyol_levante_j32_identity(self, ensemble, fixture_dicts: dict) -> None:
        """Espanyol vs Levante J32 output is byte-identical to snapshot."""
        snapshot = _load_snapshot("espanyol_levante_j32")
        feat = fixture_dicts["espanyol_levante_j32"]

        result = _prediction_to_dict(ensemble, feat)

        # Compare JSON serializations (same float precision as snapshots)
        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False)

        assert result_json == snapshot_json, (
            f"IDENTITY REGRESSION FAILURE: espanyol_levante_j32\n"
            f"Expected: {snapshot_json[:300]}\n"
            f"Got:      {result_json[:300]}"
        )

    def test_realmadrid_atletico_j29_identity(
        self, ensemble, fixture_dicts: dict
    ) -> None:
        """Real Madrid vs Atletico J29 output is byte-identical to snapshot."""
        snapshot = _load_snapshot("realmadrid_atletico_j29")
        feat = fixture_dicts["realmadrid_atletico_j29"]

        result = _prediction_to_dict(ensemble, feat)

        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False)

        assert result_json == snapshot_json, (
            f"IDENTITY REGRESSION FAILURE: realmadrid_atletico_j29\n"
            f"Expected: {snapshot_json[:300]}\n"
            f"Got:      {result_json[:300]}"
        )

    def test_villarreal_osasuna_j25_identity(
        self, ensemble, fixture_dicts: dict
    ) -> None:
        """Villarreal vs Osasuna J25 output is byte-identical to snapshot."""
        snapshot = _load_snapshot("villarreal_osasuna_j25")
        feat = fixture_dicts["villarreal_osasuna_j25"]

        result = _prediction_to_dict(ensemble, feat)

        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False)

        assert result_json == snapshot_json, (
            f"IDENTITY REGRESSION FAILURE: villarreal_osasuna_j25\n"
            f"Expected: {snapshot_json[:300]}\n"
            f"Got:      {result_json[:300]}"
        )
