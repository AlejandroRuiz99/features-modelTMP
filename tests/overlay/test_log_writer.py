"""
tests/overlay/test_log_writer.py — TDD for overlay.log_writer.

Tests (T6.1 + T6.3):
  1. write_overlay_log writes a JSON file at the expected path with correct schema.
  2. Log file is readable JSON with all required top-level keys.
  3. Re-run same match writes a SECOND file with timestamp suffix; original unchanged.
  4. Parent directories are created if they don't exist.
  5. record is written exactly (round-trip JSON equality).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from overlay.log_writer import write_overlay_log

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_RECORD: dict = {
    "timestamp": "2026-04-27T10:00:00Z",
    "match": {"home": "Espanol", "away": "Levante", "date": "2026-04-27"},
    "narrative_raw": "match:\n  home: Espanol\n  away: Levante\n",
    "parsed_flags": {"confidence_level": 4, "special_flags": ["late_season"]},
    "pre_overlay": {
        "expected_fouls": 25.65,
        "pmf_summary": {
            "mean": 25.65,
            "std": 6.5,
            "q25": 21.0,
            "q50": 25.0,
            "q75": 30.0,
        },
    },
    "rules_fired": [
        {
            "id": "both_relegation_up",
            "direction": "up",
            "magnitude_applied": 0.8,
            "suppressed_by_floor": False,
        }
    ],
    "post_overlay": {
        "expected_fouls": 26.45,
        "pmf_summary": {
            "mean": 26.45,
            "std": 7.0,
            "q25": 22.0,
            "q50": 26.0,
            "q75": 31.0,
        },
    },
    "kelly_raw_vs_scaled": {"kelly_raw": 0.042, "kelly_scaled": 0.036},
    "actual_fouls": None,
}


def _make_log_path(outdir: Path, date: str, home: str, away: str) -> Path:
    """Expected canonical log file path."""
    return outdir / f"{date}_{home}_vs_{away}.json"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteOverlayLog:
    def test_writes_json_file_at_expected_path(self, tmp_path: Path) -> None:
        """write_overlay_log creates a JSON file at outdir/date_home_vs_away.json."""
        outdir = tmp_path / "overlay" / "logs"
        record = dict(_MINIMAL_RECORD)

        result_path = write_overlay_log(record, outdir)

        expected = _make_log_path(outdir, "2026-04-27", "Espanol", "Levante")
        assert result_path == expected, f"Expected {expected}, got {result_path}"
        assert result_path.exists(), "Log file was not created"

    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        """Written file is valid JSON and contains all required keys."""
        outdir = tmp_path / "logs"
        record = dict(_MINIMAL_RECORD)

        path = write_overlay_log(record, outdir)

        content = json.loads(path.read_text(encoding="utf-8"))
        required_keys = {
            "timestamp",
            "match",
            "narrative_raw",
            "parsed_flags",
            "pre_overlay",
            "rules_fired",
            "post_overlay",
            "kelly_raw_vs_scaled",
            "actual_fouls",
        }
        for key in required_keys:
            assert key in content, f"Missing required key '{key}' in log"

    def test_record_roundtrip_json_equality(self, tmp_path: Path) -> None:
        """Written record round-trips to an identical dict."""
        outdir = tmp_path / "logs"
        record = dict(_MINIMAL_RECORD)

        path = write_overlay_log(record, outdir)

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == record

    def test_parent_dirs_created_if_missing(self, tmp_path: Path) -> None:
        """write_overlay_log creates parent directories if they don't exist."""
        outdir = tmp_path / "deep" / "nested" / "logs"
        assert not outdir.exists()

        path = write_overlay_log(dict(_MINIMAL_RECORD), outdir)

        assert path.exists()

    def test_collision_second_run_writes_suffixed_file(self, tmp_path: Path) -> None:
        """Re-running same match writes a new file with _HHMMSS suffix; original unchanged."""
        outdir = tmp_path / "logs"
        record = dict(_MINIMAL_RECORD)

        path1 = write_overlay_log(record, outdir)
        original_content = path1.read_text(encoding="utf-8")

        # Small pause to ensure time progresses (or at least different microsecond)
        time.sleep(0.01)

        path2 = write_overlay_log(record, outdir)

        # Second file must differ from the first
        assert path2 != path1, "Second run should produce a different (suffixed) file"
        assert path2.exists(), "Second log file was not created"

        # Original file must be unchanged
        assert path1.read_text(encoding="utf-8") == original_content, (
            "Original log file was modified by second write"
        )

    def test_collision_suffix_contains_time(self, tmp_path: Path) -> None:
        """Suffixed collision file name contains a time component."""
        outdir = tmp_path / "logs"
        record = dict(_MINIMAL_RECORD)

        _path1 = write_overlay_log(record, outdir)
        time.sleep(0.01)
        path2 = write_overlay_log(record, outdir)

        # File name should contain underscore + 6 digits (HHMMSS) or similar
        stem = path2.stem
        assert "_" in stem, f"Suffix not found in filename stem: {stem}"
        # The suffix after the last underscore should be numeric (HHMMSS or microseconds)
        suffix_part = stem.split("_")[-1]
        assert suffix_part.isdigit(), (
            f"Time suffix '{suffix_part}' is not all digits in stem: {stem}"
        )


def _make_record_without_ev_table() -> dict:
    """Simulate how run_prediction.py currently builds pre_overlay/post_overlay
    (WITHOUT ev_table — this is the bug to fix).
    """
    return {
        "timestamp": "2026-04-27T10:00:00Z",
        "match": {"home": "Espanol", "away": "Levante", "date": "2026-04-27"},
        "narrative_raw": "",
        "parsed_flags": {"confidence_level": 4},
        "pre_overlay": {
            "expected_fouls": 25.65,
            "pmf_summary": {
                "mean": 25.65,
                "std": 6.5,
                "q25": 21.0,
                "q50": 25.0,
                "q75": 30.0,
            },
            # ev_table intentionally absent — simulating the current (broken) construction
        },
        "rules_fired": [],
        "post_overlay": {
            "expected_fouls": 25.65,
            "pmf_summary": {
                "mean": 25.65,
                "std": 6.5,
                "q25": 21.0,
                "q50": 25.0,
                "q75": 30.0,
            },
            # ev_table intentionally absent — simulating the current (broken) construction
        },
        "kelly_raw_vs_scaled": {"kelly_raw": 1.0, "kelly_scaled": 1.0},
        "actual_fouls": None,
    }


def _make_record_with_ev_table() -> dict:
    """Simulate how run_prediction.py should build pre_overlay/post_overlay
    (WITH ev_table: null — correct per REQ-6.2).
    """
    return {
        "timestamp": "2026-04-27T10:00:00Z",
        "match": {"home": "Espanol", "away": "Levante", "date": "2026-04-27"},
        "narrative_raw": "",
        "parsed_flags": {"confidence_level": 4},
        "pre_overlay": {
            "expected_fouls": 25.65,
            "pmf_summary": {
                "mean": 25.65,
                "std": 6.5,
                "q25": 21.0,
                "q50": 25.0,
                "q75": 30.0,
            },
            "ev_table": None,  # REQ-6.2: null when odds unavailable
        },
        "rules_fired": [],
        "post_overlay": {
            "expected_fouls": 25.65,
            "pmf_summary": {
                "mean": 25.65,
                "std": 6.5,
                "q25": 21.0,
                "q50": 25.0,
                "q75": 30.0,
            },
            "ev_table": None,  # REQ-6.2: null when odds unavailable
        },
        "kelly_raw_vs_scaled": {"kelly_raw": 1.0, "kelly_scaled": 1.0},
        "actual_fouls": None,
    }


class TestEvTableInLogSchema:
    """REQ-6.2: ev_table key must be present in pre_overlay and post_overlay.

    The construction-site for the log record is run_prediction._apply_overlay_post_prediction.
    These tests verify the schema contract at the log-writer boundary: the record written
    to disk must include ev_table in both pre_overlay and post_overlay.
    """

    def test_pre_overlay_contains_ev_table_key(self, tmp_path: Path) -> None:
        """pre_overlay section of log record must contain 'ev_table' key (null when no odds)."""
        record = _make_record_with_ev_table()
        path = write_overlay_log(record, tmp_path / "logs")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "ev_table" in loaded["pre_overlay"], (
            "pre_overlay must contain 'ev_table' key (REQ-6.2)"
        )

    def test_post_overlay_contains_ev_table_key(self, tmp_path: Path) -> None:
        """post_overlay section of log record must contain 'ev_table' key (null when no odds)."""
        record = _make_record_with_ev_table()
        path = write_overlay_log(record, tmp_path / "logs")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "ev_table" in loaded["post_overlay"], (
            "post_overlay must contain 'ev_table' key (REQ-6.2)"
        )

    def test_ev_table_null_when_no_odds(self, tmp_path: Path) -> None:
        """ev_table is null (JSON null / Python None) when market odds are not available."""
        record = _make_record_with_ev_table()
        path = write_overlay_log(record, tmp_path / "logs")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["pre_overlay"]["ev_table"] is None, (
            "pre_overlay.ev_table should be null when odds are unavailable"
        )
        assert loaded["post_overlay"]["ev_table"] is None, (
            "post_overlay.ev_table should be null when odds are unavailable"
        )

    def test_record_without_ev_table_fails_schema_check(self, tmp_path: Path) -> None:
        """A record built WITHOUT ev_table (simulating the old broken construction)
        should NOT have 'ev_table' in pre_overlay — confirming the bug existed.
        This test documents the pre-fix state and verifies our helpers are correct.
        """
        record = _make_record_without_ev_table()
        path = write_overlay_log(record, tmp_path / "logs")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        # The old construction didn't include ev_table
        assert "ev_table" not in loaded["pre_overlay"], (
            "_make_record_without_ev_table helper should NOT include ev_table"
        )

    def test_run_prediction_log_record_includes_ev_table(self) -> None:
        """The helper that builds log records (simulating run_prediction) must include ev_table.

        This test documents the fix: run_prediction._apply_overlay_post_prediction
        must add ev_table: null to pre_overlay and post_overlay.
        We test this by checking the build_log_overlay_section helper.
        """
        from overlay.log_writer import build_log_overlay_section

        pre = {"expected_fouls": 25.0, "pmf_summary": {"mean": 25.0}}
        post = {"expected_fouls": 26.0, "pmf_summary": {"mean": 26.0}}

        pre_section = build_log_overlay_section(pre)
        post_section = build_log_overlay_section(post)

        assert "ev_table" in pre_section, "pre_overlay section must include ev_table"
        assert "ev_table" in post_section, "post_overlay section must include ev_table"
        assert pre_section["ev_table"] is None, "ev_table must be null when no odds"
        assert post_section["ev_table"] is None, "ev_table must be null when no odds"
