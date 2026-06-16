"""
overlay.log_writer — JSON log writer for overlay runs.

Public API:
    write_overlay_log(record, outdir) -> Path

Log file naming:
    {outdir}/{date}_{home}_vs_{away}.json

Collision handling (REQ-6.4):
    If a file already exists for the same match, a new file is written with a
    _HHMMSS suffix (e.g. date_Home_vs_Away_103045.json). If two runs happen in
    the same second, microseconds are used instead.

Atomic write:
    Writes to a temp file first, then renames atomically (POSIX rename).
    On Windows, shutil.move is used to overwrite atomically.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

__all__ = ["build_log_overlay_section", "write_overlay_log"]


def build_log_overlay_section(pmf_section: dict) -> dict:
    """Build a pre_overlay or post_overlay log section with ev_table populated.

    REQ-6.2 requires ev_table to be present in both pre_overlay and post_overlay.
    When market odds are not available at overlay time, ev_table is set to null.
    When odds are available (future extension), callers should pass the EV table list.

    Args:
        pmf_section: Dict with at minimum ``expected_fouls`` and ``pmf_summary``.
                     Any extra keys are passed through unchanged.

    Returns:
        New dict with all keys from ``pmf_section`` plus ``ev_table: None``
        (unless ``ev_table`` was already present in the input).
    """
    result = dict(pmf_section)
    if "ev_table" not in result:
        result["ev_table"] = None
    return result


def write_overlay_log(record: dict, outdir: Path) -> Path:
    """Write an overlay result record to a JSON log file.

    Args:
        record: Dict containing the full overlay log schema.  Must include
            ``match.home``, ``match.away``, and ``match.date`` sub-keys.
        outdir: Directory where the log file is written.  Created if absent.

    Returns:
        Path to the written log file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    match_info = record.get("match", {})
    date = match_info.get("date", "unknown")
    home = match_info.get("home", "home")
    away = match_info.get("away", "away")

    canonical_path = outdir / f"{date}_{home}_vs_{away}.json"

    dest_path = _resolve_path(canonical_path)

    _atomic_write(dest_path, record)
    return dest_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_path(canonical: Path) -> Path:
    """Return canonical path if it doesn't exist, else a suffixed variant.

    Suffix strategy:
      1. Try _HHMMSS from current time.
      2. If that also exists (same second), use microseconds.
    """
    if not canonical.exists():
        return canonical

    now = datetime.now()
    # First try: HHMMSS suffix
    suffix = now.strftime("%H%M%S")
    candidate = canonical.with_stem(f"{canonical.stem}_{suffix}")
    if not candidate.exists():
        return candidate

    # Second try: microseconds (6 digits)
    micro = f"{now.hour:02d}{now.minute:02d}{now.second:02d}{now.microsecond:06d}"
    return canonical.with_stem(f"{canonical.stem}_{micro}")


def _atomic_write(path: Path, record: dict) -> None:
    """Write record as JSON to path atomically via temp file + rename."""
    json_bytes = json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8")

    # Write to a sibling temp file, then rename
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, json_bytes)
        os.close(fd)
        shutil.move(tmp_path, str(path))
    except Exception:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
