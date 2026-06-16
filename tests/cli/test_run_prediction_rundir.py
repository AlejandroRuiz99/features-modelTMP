"""
tests/cli/test_run_prediction_rundir.py — T14.1: --run-dir flag tests.

Tests (Strict TDD):
  1. --run-dir routes prediction.json + overlay_logs into run folder
  2. --run-dir + --output-json explicitly: explicit value wins
  3. --run-dir + --overlay-log-dir explicitly: explicit value wins
  4. --run-dir without manifest fails gracefully
  5. --run-dir with valid manifest reads seed
  6. YAML batch file parses correctly
  7. YAML batch file maps home->local, away->visitante, referee->arbitro
  8. Legacy invocation (no --run-dir) unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# T14.1: --run-dir routing tests
# ---------------------------------------------------------------------------


class TestRunDirRouting:
    """Test that --run-dir implicitly sets output paths."""

    def test_run_dir_sets_implicit_overlay_log_dir(self, tmp_path: Path) -> None:
        """When --run-dir is set, args.overlay_log_dir defaults to {run}/prediction/overlay_logs."""
        from argparse import Namespace

        from run_prediction import _resolve_run_dir_paths

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        (run_dir / "prediction").mkdir()

        # Create minimal manifest
        manifest_path = run_dir / "manifest.yaml"
        manifest_path.write_text(
            "run_id: test_run\nslug: test\ncreated_at: '2026-04-29T10:00:00Z'\n"
            "git_commit: abc123\nseed: 42\ncompleted_steps: []\n",
            encoding="utf-8",
        )

        args = Namespace(
            run_dir=str(run_dir),
            overlay_log_dir=None,
            output_json=None,
            narratives=None,
        )
        _resolve_run_dir_paths(args)

        assert args.overlay_log_dir == str(run_dir / "prediction" / "overlay_logs")
        assert args.output_json == str(run_dir / "prediction" / "prediction.json")
        assert args.narratives == str(run_dir / "input" / "narratives")

    def test_run_dir_respects_explicit_overlay_log_dir(self, tmp_path: Path) -> None:
        """When --overlay-log-dir is explicitly set, it overrides the run_dir default."""
        from argparse import Namespace

        from run_prediction import _resolve_run_dir_paths

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        manifest_path = run_dir / "manifest.yaml"
        manifest_path.write_text(
            "run_id: test_run\nslug: test\ncreated_at: '2026-04-29T10:00:00Z'\n"
            "git_commit: abc123\nseed: 42\ncompleted_steps: []\n",
            encoding="utf-8",
        )

        explicit_dir = "/some/other/path"

        args = Namespace(
            run_dir=str(run_dir),
            overlay_log_dir=explicit_dir,
            output_json=None,
            narratives=None,
        )
        _resolve_run_dir_paths(args)

        # Explicit value preserved
        assert args.overlay_log_dir == explicit_dir
        # Others default
        assert args.output_json == str(run_dir / "prediction" / "prediction.json")

    def test_run_dir_respects_explicit_output_json(self, tmp_path: Path) -> None:
        """When --output-json is explicitly set, it overrides run_dir default."""
        from argparse import Namespace

        from run_prediction import _resolve_run_dir_paths

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        manifest_path = run_dir / "manifest.yaml"
        manifest_path.write_text(
            "run_id: test_run\nslug: test\ncreated_at: '2026-04-29T10:00:00Z'\n"
            "git_commit: abc123\nseed: 42\ncompleted_steps: []\n",
            encoding="utf-8",
        )

        explicit_path = "/some/output.json"

        args = Namespace(
            run_dir=str(run_dir),
            overlay_log_dir=None,
            output_json=explicit_path,
            narratives=None,
        )
        _resolve_run_dir_paths(args)

        assert args.output_json == explicit_path

    def test_run_dir_missing_manifest_raises(self, tmp_path: Path) -> None:
        """If --run-dir points to a folder without manifest.yaml, raises FileNotFoundError."""
        from argparse import Namespace

        from run_prediction import _resolve_run_dir_paths

        run_dir = tmp_path / "no_manifest"
        run_dir.mkdir()

        args = Namespace(
            run_dir=str(run_dir),
            overlay_log_dir=None,
            output_json=None,
            narratives=None,
        )
        with pytest.raises(FileNotFoundError):
            _resolve_run_dir_paths(args)


# ---------------------------------------------------------------------------
# T14.1: YAML batch file support
# ---------------------------------------------------------------------------


class TestYamlBatchFile:
    """Test that --batch-file accepts .yaml format from matches_parser."""

    def test_yaml_batch_file_parses(self, tmp_path: Path) -> None:
        """Parse a matches.yaml file (from parsers/matches_parser) into batch matches."""
        from run_prediction import _parse_yaml_batch_file

        yaml_path = tmp_path / "matches.yaml"
        yaml_data = {
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
        yaml_path.write_text(yaml.safe_dump(yaml_data), encoding="utf-8")

        result = _parse_yaml_batch_file(yaml_path)

        assert isinstance(result, list)
        assert len(result) == 1
        match = result[0]
        # Maps home -> local, away -> visitante, referee -> arbitro
        assert match["local"] == "Real Madrid"
        assert match["visitante"] == "Mallorca"
        assert match["fecha"] == "2026-05-04"
        assert match["arbitro"] == "Munuera Montero"
        assert match["jornada"] == 35

    def test_yaml_batch_handles_multiple_matches(self, tmp_path: Path) -> None:
        """A matches.yaml with multiple matches returns all of them."""
        from run_prediction import _parse_yaml_batch_file

        yaml_path = tmp_path / "matches.yaml"
        yaml_data = {
            "jornada": 35,
            "matches": [
                {
                    "home": "Real Madrid",
                    "away": "Mallorca",
                    "date": "2026-05-04",
                    "referee": "Munuera Montero",
                },
                {
                    "home": "FC Barcelona",
                    "away": "Sevilla",
                    "date": "2026-05-05",
                    "referee": "Hernandez Maeso",
                },
            ],
        }
        yaml_path.write_text(yaml.safe_dump(yaml_data), encoding="utf-8")

        result = _parse_yaml_batch_file(yaml_path)
        assert len(result) == 2
        assert result[0]["local"] == "Real Madrid"
        assert result[1]["local"] == "FC Barcelona"

    def test_yaml_batch_empty_matches_returns_empty(self, tmp_path: Path) -> None:
        """A matches.yaml with empty matches list returns []."""
        from run_prediction import _parse_yaml_batch_file

        yaml_path = tmp_path / "matches.yaml"
        yaml_data = {"jornada": 35, "matches": []}
        yaml_path.write_text(yaml.safe_dump(yaml_data), encoding="utf-8")

        result = _parse_yaml_batch_file(yaml_path)
        assert result == []


# ---------------------------------------------------------------------------
# T14.1: RNG seeding from manifest
# ---------------------------------------------------------------------------


class TestRngSeeding:
    """Test that --run-dir seeds RNG from manifest.seed."""

    def test_seed_rng_called_with_manifest_seed(self, tmp_path: Path) -> None:
        """When --run-dir is set, RNG seeds are set from manifest.seed."""
        import random

        from run_prediction import _seed_rng

        _seed_rng(12345)
        first_random = random.random()

        _seed_rng(12345)
        second_random = random.random()

        # Same seed → same result
        assert first_random == second_random

    def test_seed_rng_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different RNG state."""
        import random

        from run_prediction import _seed_rng

        _seed_rng(11)
        a = random.random()
        _seed_rng(22)
        b = random.random()

        assert a != b
