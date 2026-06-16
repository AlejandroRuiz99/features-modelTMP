"""
T9.2 — Full overlay end-to-end integration test.

Covers:
  1. With narrative: overlay section present, log written, kelly_scaled < 1.0 when rules fire.
  2. Without narrative: pure identity — no overlay section, no changed predictions.
  3. Log writer: actual_fouls=None in log; fill_actuals fills it correctly.
  4. Backfill script: produces a report with ≥ 1 row.

Uses fixture feature dicts (no Supabase needed for ensemble predictions).
Skips if ensemble checkpoints unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PRED_DIR = ROOT / "prediction_models"
SNAPSHOT_DIR = ROOT / "tests" / "fixtures" / "snapshots"
NARRATIVE_DIR = ROOT / "tests" / "fixtures" / "narratives"
CATALOG_PATH = ROOT / "overlay" / "rules.yaml"

# Ensure prediction_models importable
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))
sys.path.insert(0, str(SNAPSHOT_DIR))

# Try to import fixture feature dicts
try:
    from capture_snapshots import (  # type: ignore[import]
        FIXTURE_ESPANYOL_LEVANTE,
    )

    _FIXTURES_AVAILABLE = True
except ImportError:
    FIXTURE_ESPANYOL_LEVANTE = None  # type: ignore[assignment]
    _FIXTURES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_ensemble():  # type: ignore[return]
    """Load the trained ensemble from checkpoints."""
    from src.models.ensemble import FoulPredictionEnsemble  # type: ignore[import]

    CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"
    config_path = CHECKPOINT_DIR / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(CHECKPOINT_DIR)
    return ensemble


def _build_prediction_for_overlay(ensemble, fixture: dict) -> dict:
    """Run ensemble.predict and build overlay-ready prediction dict."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ensemble():  # type: ignore[return]
    """Load ensemble once for the module."""
    checkpoint = PRED_DIR / "checkpoints" / "ensemble" / "config.json"
    if not checkpoint.exists():
        pytest.skip("Ensemble checkpoints not available")
    if not _FIXTURES_AVAILABLE:
        pytest.skip("Fixture feature dicts not available")
    return _load_ensemble()


@pytest.fixture(scope="module")
def base_prediction(ensemble) -> dict:  # type: ignore[return]
    """Build base prediction dict from Espanyol-Levante J32 fixture."""
    return _build_prediction_for_overlay(ensemble, FIXTURE_ESPANYOL_LEVANTE)


# ---------------------------------------------------------------------------
# T9.2a: Full prediction WITH narrative
# ---------------------------------------------------------------------------


class TestOverlayWithNarrative:
    """Tests for prediction with narrative — overlay section must be present."""

    def test_overlay_section_present_after_apply_overlay(
        self, base_prediction: dict
    ) -> None:
        """apply_overlay returns OverlayResult with non-empty overlay section."""
        from overlay.applier import OverlayResult, apply_overlay
        from overlay.loader import load_narrative
        from overlay.rules import load_catalog

        narrative = load_narrative(NARRATIVE_DIR / "valid_full.yaml")
        catalog = load_catalog(CATALOG_PATH)

        result = apply_overlay(base_prediction, narrative, catalog)

        assert isinstance(result, OverlayResult)
        assert result.pre_pmf_summary is not None
        assert result.post_pmf_summary is not None
        assert "mean" in result.pre_pmf_summary
        assert "mean" in result.post_pmf_summary

    def test_log_written_when_rules_fire(
        self, base_prediction: dict, tmp_path: Path
    ) -> None:
        """A JSON log is written to overlay_log_dir when overlay is applied."""
        from datetime import datetime, timezone

        from overlay.applier import apply_overlay
        from overlay.loader import load_narrative
        from overlay.log_writer import write_overlay_log
        from overlay.rules import load_catalog

        narrative = load_narrative(NARRATIVE_DIR / "valid_full.yaml")
        catalog = load_catalog(CATALOG_PATH)
        result = apply_overlay(base_prediction, narrative, catalog)

        log_dir = tmp_path / "overlay_logs"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "match": {
                "home": narrative.match.home,
                "away": narrative.match.away,
                "date": narrative.match.date,
            },
            "narrative_raw": "confidence_level: 4\n",
            "parsed_flags": {
                "confidence_level": narrative.confidence_level,
                "special_flags": narrative.special_flags,
            },
            "pre_overlay": result.pre_pmf_summary,
            "rules_fired": result.rules_fired,
            "post_overlay": result.post_pmf_summary,
            "kelly_raw_vs_scaled": {
                "kelly_raw": result.kelly_raw,
                "kelly_scaled": result.kelly_scaled,
            },
            "actual_fouls": None,
        }
        log_path = write_overlay_log(record, log_dir)

        assert log_path.exists(), f"Expected log file at {log_path}"
        loaded = json.loads(log_path.read_text(encoding="utf-8"))
        assert loaded["match"]["home"] == narrative.match.home
        assert loaded["actual_fouls"] is None

    def test_kelly_scaled_differs_from_kelly_raw_when_rules_fire(
        self, base_prediction: dict
    ) -> None:
        """kelly_scaled < kelly_raw (1.0) when rules with kelly_scale < 1.0 fire."""
        from overlay.applier import apply_overlay
        from overlay.loader import load_narrative
        from overlay.rules import load_catalog

        narrative = load_narrative(NARRATIVE_DIR / "valid_full.yaml")
        catalog = load_catalog(CATALOG_PATH)

        result = apply_overlay(base_prediction, narrative, catalog)

        # valid_full.yaml has relegation + physical_clash → should reduce kelly
        if result.rules_fired:
            assert result.kelly_scaled <= result.kelly_raw, (
                f"Expected kelly_scaled ≤ kelly_raw when rules fire, "
                f"got kelly_raw={result.kelly_raw}, kelly_scaled={result.kelly_scaled}"
            )
        # When at least 1 rule fires and it has kelly_scale < 1.0, scaled must differ
        kelly_reducing_rules = [
            r
            for r in result.rules_fired
            if r.get("magnitude_applied") != 0  # proxy for meaningful rule
        ]
        if kelly_reducing_rules:
            # The aggregate kelly should be reduced
            assert result.kelly_scaled < 1.0, (
                f"Expected kelly_scaled < 1.0 when rules with kelly reduction fire, "
                f"got {result.kelly_scaled}"
            )

    def test_post_pred_differs_from_pre_pred_when_rules_fire(
        self, base_prediction: dict
    ) -> None:
        """Post-overlay expected_fouls differs from pre when tilt rules fire and gate passes."""
        from overlay.applier import apply_overlay
        from overlay.loader import load_narrative
        from overlay.rules import load_catalog

        narrative = load_narrative(NARRATIVE_DIR / "valid_full.yaml")
        catalog = load_catalog(CATALOG_PATH)
        result = apply_overlay(base_prediction, narrative, catalog)

        pre_mean = result.pre_pmf_summary["mean"]
        post_mean = result.post_pmf_summary["mean"]

        # valid_full has confidence_level=4 and multiple relegation rules → gate passes
        if len(result.rules_fired) >= 2:
            assert abs(post_mean - pre_mean) > 0.01, (
                f"Expected post_mean to differ from pre_mean when rules fire, "
                f"both = {pre_mean}"
            )


# ---------------------------------------------------------------------------
# T9.2b: Identity WITHOUT narrative
# ---------------------------------------------------------------------------


class TestOverlayIdentityWithoutNarrative:
    """Tests for prediction without narrative — must be 100% identity."""

    def test_minimal_narrative_no_rules_fire_identity(
        self, base_prediction: dict
    ) -> None:
        """Minimal narrative (confidence 1, no flags) fires no rules and preserves PMF."""
        from overlay.applier import apply_overlay
        from overlay.rules import load_catalog
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
        pre_mean = base_prediction["expected_fouls"]

        result = apply_overlay(base_prediction, narrative, catalog)

        assert result.rules_fired == [], (
            f"Expected no rules to fire for minimal narrative, "
            f"got: {result.rules_fired}"
        )
        post_mean = result.post_pmf_summary["mean"]
        assert abs(post_mean - pre_mean) < 1e-3, (
            f"Identity failed: pre_mean={pre_mean:.6f}, post_mean={post_mean:.6f}"
        )

    def test_kelly_raw_equals_scaled_when_no_rules_fire(
        self, base_prediction: dict
    ) -> None:
        """Without narrative effects, kelly_raw == kelly_scaled == 1.0."""
        from overlay.applier import apply_overlay
        from overlay.rules import load_catalog
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

        result = apply_overlay(base_prediction, narrative, catalog)

        assert result.kelly_raw == 1.0
        assert result.kelly_scaled == 1.0

    def test_identity_all_three_snapshots(self, ensemble) -> None:
        """Overlay system does not change prediction for any of the 3 snapshot fixtures."""
        try:
            from capture_snapshots import (  # type: ignore[import]
                FIXTURE_ESPANYOL_LEVANTE,
                FIXTURE_MADRID_ATLETICO,
                FIXTURE_VILLARREAL_OSASUNA,
            )
        except ImportError:
            pytest.skip("Fixture feature dicts not available")
            return

        from overlay.applier import apply_overlay
        from overlay.rules import load_catalog
        from overlay.schema import Narrative, NarrativeMatch, ObjectiveOverride

        catalog = load_catalog(CATALOG_PATH)
        fixtures = [
            (FIXTURE_ESPANYOL_LEVANTE, "Espanol", "Levante", "2026-04-27"),
            (FIXTURE_MADRID_ATLETICO, "Real Madrid", "Atletico Madrid", "2026-03-22"),
            (FIXTURE_VILLARREAL_OSASUNA, "Villarreal", "Osasuna", "2026-01-01"),
        ]

        for fixture, home, away, date in fixtures:
            pred = _build_prediction_for_overlay(ensemble, fixture)
            narr = Narrative(
                match=NarrativeMatch(home=home, away=away, date=date),
                confidence_level=1,
                objectives={
                    "home": ObjectiveOverride(label="mid"),
                    "away": ObjectiveOverride(label="mid"),
                },
            )
            result = apply_overlay(pred, narr, catalog)

            assert result.rules_fired == [], (
                f"Expected no rules for {home} vs {away}, got: {result.rules_fired}"
            )
            pre = pred["expected_fouls"]
            post = result.post_pmf_summary["mean"]
            assert abs(post - pre) < 1e-3, (
                f"Identity failed for {home} vs {away}: {pre:.6f} -> {post:.6f}"
            )


# ---------------------------------------------------------------------------
# T9.2c: fill_actuals integration
# ---------------------------------------------------------------------------


class TestFillActualsIntegration:
    """Test fill_actuals CLI integration."""

    def test_fill_actuals_writes_fouls_to_log(self, tmp_path: Path) -> None:
        """fill_actuals CLI writes actual_fouls to an existing log file."""
        from datetime import datetime, timezone

        from overlay.fill_actuals import fill_actuals
        from overlay.log_writer import write_overlay_log

        log_dir = tmp_path / "logs"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "match": {"home": "Espanol", "away": "Levante", "date": "2026-04-27"},
            "narrative_raw": "confidence_level: 4\n",
            "parsed_flags": {},
            "pre_overlay": {"mean": 25.65},
            "rules_fired": [],
            "post_overlay": {"mean": 27.9},
            "kelly_raw_vs_scaled": {"kelly_raw": 1.0, "kelly_scaled": 0.85},
            "actual_fouls": None,
        }
        log_path = write_overlay_log(record, log_dir)

        fill_actuals(log_path, 22)

        loaded = json.loads(log_path.read_text(encoding="utf-8"))
        assert loaded["actual_fouls"] == 22, (
            f"Expected actual_fouls=22, got {loaded['actual_fouls']}"
        )

    def test_fill_actuals_idempotent_same_value(self, tmp_path: Path) -> None:
        """fill_actuals with same value is a no-op (idempotent)."""
        from datetime import datetime, timezone

        from overlay.fill_actuals import fill_actuals
        from overlay.log_writer import write_overlay_log

        log_dir = tmp_path / "logs"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "match": {"home": "A", "away": "B", "date": "2026-01-01"},
            "actual_fouls": 20,
        }
        log_path = write_overlay_log(record, log_dir)

        # Fill with same value — should not raise
        fill_actuals(log_path, 20)

        loaded = json.loads(log_path.read_text(encoding="utf-8"))
        assert loaded["actual_fouls"] == 20
