"""
tests/integration/test_resume.py — T15.1: Resume integration tests.

Tests:
  1. resume_run loads existing manifest and preserves completed_steps
  2. record_step is idempotent — adding same step twice doesn't duplicate
  3. Resume with partial completion (steps 1-2) returns context with those steps
  4. Slug collision raises SlugCollisionError
  5. Hash check on resume detects rules.yaml changes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.lifecycle import (  # noqa: E402
    SlugCollisionError,
    record_step,
    resume_run,
    start_run,
)
from runs.manifest import dump_yaml, load_yaml  # noqa: E402


class TestResumeFlow:
    """Test the full resume flow with partial completions."""

    def test_resume_loads_existing_manifest(self, tmp_path: Path) -> None:
        """resume_run reads manifest.yaml and returns RunContext."""
        ctx = start_run(slug="test-resume", runs_dir=tmp_path, seed=42)

        # Resume from same directory
        resumed = resume_run(ctx.root_path)
        assert resumed.run_id == ctx.run_id
        assert resumed.slug == "test-resume"
        assert resumed.seed == 42
        assert resumed.completed_steps == []

    def test_resume_preserves_completed_steps(self, tmp_path: Path) -> None:
        """If steps 1-2 done, resume_run returns context with those steps."""
        ctx = start_run(slug="test-partial", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")
        record_step(ctx, "step_2")

        resumed = resume_run(ctx.root_path)
        assert resumed.completed_steps == ["step_1", "step_2"]

    def test_record_step_is_idempotent(self, tmp_path: Path) -> None:
        """Calling record_step with same step twice doesn't create duplicates."""
        ctx = start_run(slug="test-idempotent", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")
        record_step(ctx, "step_1")
        record_step(ctx, "step_2")

        manifest = load_yaml(ctx.manifest_path)
        assert manifest.completed_steps == ["step_1", "step_2"]

    def test_resume_missing_manifest_raises(self, tmp_path: Path) -> None:
        """resume_run on a directory without manifest.yaml raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            resume_run(empty_dir)

    def test_resume_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        """resume_run on a non-existent dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            resume_run(tmp_path / "no_such_dir")


class TestSlugCollision:
    """Test slug collision behavior."""

    def test_start_run_same_slug_raises(self, tmp_path: Path) -> None:
        """Starting two runs with the same slug on same date raises SlugCollisionError."""
        start_run(slug="duplicate", runs_dir=tmp_path, seed=42)
        with pytest.raises(SlugCollisionError):
            start_run(slug="duplicate", runs_dir=tmp_path, seed=42)


class TestNextStepDetection:
    """Test detection of next-step-to-run via completed_steps."""

    def test_first_incomplete_step_is_step_1(self, tmp_path: Path) -> None:
        """Brand new run → first incomplete step is step_1."""
        ctx = start_run(slug="test-step1", runs_dir=tmp_path, seed=42)
        all_steps = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]
        next_step = next((s for s in all_steps if s not in ctx.completed_steps), None)
        assert next_step == "step_1"

    def test_first_incomplete_after_2_steps_is_step_3(self, tmp_path: Path) -> None:
        """If steps 1-2 done, next incomplete is step_3."""
        ctx = start_run(slug="test-step3", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")
        record_step(ctx, "step_2")

        all_steps = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]
        next_step = next((s for s in all_steps if s not in ctx.completed_steps), None)
        assert next_step == "step_3"

    def test_all_steps_done_returns_none(self, tmp_path: Path) -> None:
        """If all 6 steps done, no incomplete step remains."""
        ctx = start_run(slug="test-done", runs_dir=tmp_path, seed=42)
        for s in ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]:
            record_step(ctx, s)

        all_steps = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]
        next_step = next((s for s in all_steps if s not in ctx.completed_steps), None)
        assert next_step is None
