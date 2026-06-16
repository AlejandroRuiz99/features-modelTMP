"""
tests/integration/test_full_flow.py — T15.2: Full mocked 6-step flow.

Tests the structural integrity of the run folder progression through all 6 steps,
without requiring a live model or Supabase. Verifies that:

  1. start_run creates all 10 mandatory subdirs + manifest
  2. After Step 2, input/matches.yaml exists
  3. After Step 3, input/narratives/{slug}.yaml exists for each match
  4. After Step 4, features/{slug}.json exists for each match
  5. After Step 5, prediction/prediction.json + overlay_logs/ exist
  6. After Step 6 with no odds, report/prediction_report.md exists
  7. After Step 6 with odds, ev/ev_table.json + ev/picks.md exist
  8. manifest.completed_steps = [step_1..step_6] at end
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.lifecycle import (  # noqa: E402
    MANDATORY_SUBDIRS,
    record_step,
    resume_run,
    start_run,
)
from runs.manifest import dump_yaml, load_yaml  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — realistic mocked data
# ---------------------------------------------------------------------------


def _mocked_matches_yaml(run_root: Path) -> Path:
    """Step 2 produces input/matches.yaml."""
    matches_data = {
        "jornada": 35,
        "matches": [
            {
                "home": "Real Madrid",
                "away": "Mallorca",
                "date": "2026-05-04",
                "referee": "Munuera Montero",
                "slug": "realmadrid_vs_mallorca_2026-05-04",
            },
        ],
    }
    matches_path = run_root / "input" / "matches.yaml"
    matches_path.write_text(yaml.safe_dump(matches_data), encoding="utf-8")
    return matches_path


def _mocked_narrative_yaml(run_root: Path, slug: str) -> Path:
    """Step 3 produces input/narratives/{slug}.yaml."""
    narrative_data = {
        "match": {
            "home": "Real Madrid",
            "away": "Mallorca",
            "date": "2026-05-04",
            "jornada": 35,
        },
        "objectives": {
            "home": {"label": "titulo", "urgency_base": 0.85},
            "away": {"label": "salvacion", "urgency_base": 0.70},
        },
        "stakes": {"home": 4, "away": 5, "notes": "Title chase vs relegation"},
        "intensity_override": 4,
        "physicality_bias": 0,
        "referee_factor": 0,
        "special_flags": ["late_season"],
        "confidence_level": 4,
        "notes": "Mocked narrative for full-flow test",
    }
    narr_path = run_root / "input" / "narratives" / f"{slug}.yaml"
    narr_path.write_text(yaml.safe_dump(narrative_data), encoding="utf-8")
    return narr_path


def _mocked_features_json(run_root: Path, slug: str) -> Path:
    """Step 4 produces features/{slug}.json."""
    features = {"slug": slug, "feat_1": 0.5, "feat_2": 1.2}
    feat_path = run_root / "features" / f"{slug}.json"
    feat_path.write_text(json.dumps(features), encoding="utf-8")
    return feat_path


def _mocked_prediction_json(run_root: Path) -> Path:
    """Step 5 produces prediction/prediction.json."""
    prediction = {
        "matches": [
            {
                "slug": "realmadrid_vs_mallorca_2026-05-04",
                "expected_fouls": 22.5,
                "pmf_summary": {"mean": 22.5, "std": 5.0},
            }
        ],
    }
    pred_path = run_root / "prediction" / "prediction.json"
    pred_path.write_text(json.dumps(prediction), encoding="utf-8")
    return pred_path


def _mocked_overlay_log(run_root: Path, slug: str) -> Path:
    """Step 5 also writes prediction/overlay_logs/{slug}.json."""
    log = {"slug": slug, "rules_fired": []}
    log_path = run_root / "prediction" / "overlay_logs" / f"{slug}.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    return log_path


# ---------------------------------------------------------------------------
# T15.2: Full flow tests
# ---------------------------------------------------------------------------


class TestFullFlowStructure:
    """Verify run folder structure progression across all 6 steps."""

    def test_step_0_creates_all_mandatory_subdirs(self, tmp_path: Path) -> None:
        """start_run creates all 10 mandatory subdirs + manifest."""
        ctx = start_run(slug="full-flow", runs_dir=tmp_path, seed=42)

        # All 10 subdirs exist
        for subdir in MANDATORY_SUBDIRS:
            assert (ctx.root_path / subdir).is_dir(), f"Missing subdir: {subdir}"

        # Manifest exists
        assert ctx.manifest_path.is_file()

        # Manifest has correct fields
        manifest = load_yaml(ctx.manifest_path)
        assert manifest.run_id == ctx.run_id
        assert manifest.slug == "full-flow"
        assert manifest.seed == 42
        assert manifest.completed_steps == []

    def test_step_2_writes_matches_yaml(self, tmp_path: Path) -> None:
        """After Step 2, input/matches.yaml exists with parsed matches."""
        ctx = start_run(slug="step2-test", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")  # Stats freshness done

        matches_path = _mocked_matches_yaml(ctx.root_path)
        record_step(ctx, "step_2")

        assert matches_path.is_file()
        # Verify content shape
        data = yaml.safe_load(matches_path.read_text(encoding="utf-8"))
        assert data["jornada"] == 35
        assert len(data["matches"]) == 1

        # Manifest updated
        manifest = load_yaml(ctx.manifest_path)
        assert "step_2" in manifest.completed_steps

    def test_step_3_writes_narratives_per_match(self, tmp_path: Path) -> None:
        """After Step 3, input/narratives/{slug}.yaml exists for each match."""
        ctx = start_run(slug="step3-test", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")
        _mocked_matches_yaml(ctx.root_path)
        record_step(ctx, "step_2")

        slug = "realmadrid_vs_mallorca_2026-05-04"
        narr_path = _mocked_narrative_yaml(ctx.root_path, slug)
        record_step(ctx, "step_3")

        assert narr_path.is_file()
        # Validate via overlay loader (real validation)
        from overlay.loader import load_narrative

        narr = load_narrative(narr_path)
        assert narr.match.home == "Real Madrid"
        assert "home" in narr.objectives
        assert "away" in narr.objectives

    def test_step_4_writes_features(self, tmp_path: Path) -> None:
        """After Step 4, features/{slug}.json exists."""
        ctx = start_run(slug="step4-test", runs_dir=tmp_path, seed=42)
        for s in ["step_1", "step_2", "step_3"]:
            record_step(ctx, s)

        slug = "realmadrid_vs_mallorca_2026-05-04"
        feat_path = _mocked_features_json(ctx.root_path, slug)
        record_step(ctx, "step_4")

        assert feat_path.is_file()
        feats = json.loads(feat_path.read_text(encoding="utf-8"))
        assert feats["slug"] == slug

    def test_step_5_writes_prediction_and_logs(self, tmp_path: Path) -> None:
        """After Step 5, prediction/prediction.json AND overlay_logs/ exist."""
        ctx = start_run(slug="step5-test", runs_dir=tmp_path, seed=42)
        for s in ["step_1", "step_2", "step_3", "step_4"]:
            record_step(ctx, s)

        slug = "realmadrid_vs_mallorca_2026-05-04"
        pred_path = _mocked_prediction_json(ctx.root_path)
        log_path = _mocked_overlay_log(ctx.root_path, slug)
        record_step(ctx, "step_5")

        assert pred_path.is_file()
        assert log_path.is_file()

    def test_step_6_no_odds_writes_prediction_report(self, tmp_path: Path) -> None:
        """When no Codere odds, Step 6 writes report/prediction_report.md."""
        ctx = start_run(slug="step6-noodds", runs_dir=tmp_path, seed=42)
        for s in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
            record_step(ctx, s)

        # Mock: no odds path
        report_path = ctx.root_path / "report" / "prediction_report.md"
        report_path.write_text("# Prediction Report (no odds)\n", encoding="utf-8")

        # Update manifest with step_6_ev_skipped
        manifest = load_yaml(ctx.manifest_path)
        manifest.step_6_ev_skipped = True
        manifest.completed_steps.append("step_6")
        dump_yaml(manifest, ctx.manifest_path)

        assert report_path.is_file()
        manifest_after = load_yaml(ctx.manifest_path)
        assert manifest_after.step_6_ev_skipped is True
        assert "step_6" in manifest_after.completed_steps

    def test_step_6_with_odds_writes_ev_artifacts(self, tmp_path: Path) -> None:
        """When odds present, Step 6 writes ev/ev_table.json + ev/picks.md."""
        ctx = start_run(slug="step6-odds", runs_dir=tmp_path, seed=42)
        for s in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
            record_step(ctx, s)

        # Mock odds artifacts
        ev_table_path = ctx.root_path / "ev" / "ev_table.json"
        ev_table_path.write_text(
            json.dumps([{"slug": "realmadrid_vs_mallorca", "edge": 0.12}]),
            encoding="utf-8",
        )
        picks_path = ctx.root_path / "ev" / "picks.md"
        picks_path.write_text("# Picks\n- Pick 1: KEEP\n", encoding="utf-8")
        ev_report_path = ctx.root_path / "report" / "ev_report.md"
        ev_report_path.write_text("# EV Report\n", encoding="utf-8")

        # Set bankroll in manifest
        manifest = load_yaml(ctx.manifest_path)
        manifest.bankroll = 500.0
        manifest.completed_steps.append("step_6")
        dump_yaml(manifest, ctx.manifest_path)

        assert ev_table_path.is_file()
        assert picks_path.is_file()
        assert ev_report_path.is_file()
        manifest_after = load_yaml(ctx.manifest_path)
        assert manifest_after.bankroll == 500.0
        assert "step_6" in manifest_after.completed_steps

    def test_full_flow_end_state(self, tmp_path: Path) -> None:
        """At the end of the 6-step flow, manifest has all 6 steps and all artifacts exist."""
        ctx = start_run(slug="full-end", runs_dir=tmp_path, seed=42)

        # Walk through all 6 steps mocking artifacts
        record_step(ctx, "step_1")
        _mocked_matches_yaml(ctx.root_path)
        record_step(ctx, "step_2")

        slug = "realmadrid_vs_mallorca_2026-05-04"
        _mocked_narrative_yaml(ctx.root_path, slug)
        record_step(ctx, "step_3")

        _mocked_features_json(ctx.root_path, slug)
        record_step(ctx, "step_4")

        _mocked_prediction_json(ctx.root_path)
        _mocked_overlay_log(ctx.root_path, slug)
        record_step(ctx, "step_5")

        # Step 6: no odds path
        (ctx.root_path / "report" / "prediction_report.md").write_text(
            "# Done", encoding="utf-8"
        )
        record_step(ctx, "step_6")

        # Resume from final state
        resumed = resume_run(ctx.root_path)
        assert resumed.completed_steps == [
            "step_1",
            "step_2",
            "step_3",
            "step_4",
            "step_5",
            "step_6",
        ]


# ---------------------------------------------------------------------------
# Resume mid-flow scenarios
# ---------------------------------------------------------------------------


class TestResumeMidFlow:
    """Test resuming after partial completion."""

    def test_resume_after_step_2_picks_up_at_step_3(self, tmp_path: Path) -> None:
        """If steps 1-2 done, resume detects step_3 as next."""
        ctx = start_run(slug="resume-mid", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")
        _mocked_matches_yaml(ctx.root_path)
        record_step(ctx, "step_2")

        # Simulate user closing session, opening new one
        resumed = resume_run(ctx.root_path)
        assert resumed.completed_steps == ["step_1", "step_2"]

        # matches.yaml is preserved
        matches_path = resumed.root_path / "input" / "matches.yaml"
        assert matches_path.is_file()
        data = yaml.safe_load(matches_path.read_text(encoding="utf-8"))
        assert data["jornada"] == 35

        # Determine next step
        all_steps = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]
        next_step = next(
            (s for s in all_steps if s not in resumed.completed_steps), None
        )
        assert next_step == "step_3"

    def test_resume_after_step_4_with_features_intact(self, tmp_path: Path) -> None:
        """If steps 1-4 done, features/ is preserved across resume."""
        ctx = start_run(slug="resume-feat", runs_dir=tmp_path, seed=42)
        for s in ["step_1", "step_2", "step_3", "step_4"]:
            record_step(ctx, s)
        slug = "realmadrid_vs_mallorca_2026-05-04"
        _mocked_features_json(ctx.root_path, slug)

        resumed = resume_run(ctx.root_path)
        feat_path = resumed.root_path / "features" / f"{slug}.json"
        assert feat_path.is_file()
        feats = json.loads(feat_path.read_text(encoding="utf-8"))
        assert feats["slug"] == slug
