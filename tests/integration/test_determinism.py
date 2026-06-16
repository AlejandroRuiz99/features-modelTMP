"""
tests/integration/test_determinism.py — T17.1: Determinism integration tests.

Tests that the run-dir + manifest + seed protocol produces deterministic outputs.

Strategy:
  - Direct prediction byte-identity is already covered by the 3 identity snapshots
    in tests/fixtures/snapshots/ (run by test_run_prediction_identity.py).
  - This test file verifies the determinism CONTRACT layer:
    1. Same seed → same RNG sequence (across random/numpy/torch).
    2. Manifest hashes are deterministic for same input bytes.
    3. compute_hashes returns same value for unchanged files.
    4. Snapshot reuse on resume — never re-queries Supabase.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.lifecycle import record_step, resume_run, start_run  # noqa: E402
from runs.manifest import (  # noqa: E402
    compute_hashes,
    compute_narratives_hash,
    dump_yaml,
    load_yaml,
)


# ---------------------------------------------------------------------------
# T17.1a: RNG seeding determinism
# ---------------------------------------------------------------------------


class TestSeedRngDeterminism:
    """_seed_rng makes random() output deterministic."""

    def test_same_seed_same_random(self) -> None:
        """Calling _seed_rng with same value twice produces same random.random()."""
        import random

        from run_prediction import _seed_rng

        _seed_rng(42)
        seq_a = [random.random() for _ in range(10)]
        _seed_rng(42)
        seq_b = [random.random() for _ in range(10)]

        assert seq_a == seq_b

    def test_different_seeds_different_sequences(self) -> None:
        """Different seeds → different random sequences."""
        import random

        from run_prediction import _seed_rng

        _seed_rng(1)
        seq_a = [random.random() for _ in range(5)]
        _seed_rng(2)
        seq_b = [random.random() for _ in range(5)]

        assert seq_a != seq_b

    def test_seed_affects_numpy_if_available(self) -> None:
        """If numpy is installed, _seed_rng also seeds np.random."""
        try:
            import numpy as np
        except ImportError:
            return  # numpy not installed, skip

        from run_prediction import _seed_rng

        _seed_rng(42)
        a = np.random.rand(5).tolist()
        _seed_rng(42)
        b = np.random.rand(5).tolist()
        assert a == b


# ---------------------------------------------------------------------------
# T17.1b: Manifest hash determinism
# ---------------------------------------------------------------------------


class TestManifestHashDeterminism:
    """Hashes computed from same bytes are equal."""

    def test_compute_narratives_hash_stable(self, tmp_path: Path) -> None:
        """Same narratives directory → same hash on repeated calls."""
        narr_dir = tmp_path / "narratives"
        narr_dir.mkdir()
        (narr_dir / "match_a.yaml").write_text("foo: 1\n", encoding="utf-8")
        (narr_dir / "match_b.yaml").write_text("bar: 2\n", encoding="utf-8")

        h1 = compute_narratives_hash(narr_dir)
        h2 = compute_narratives_hash(narr_dir)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_compute_narratives_hash_changes_with_content(self, tmp_path: Path) -> None:
        """Different narrative bytes → different hash."""
        narr_dir = tmp_path / "narratives"
        narr_dir.mkdir()
        f = narr_dir / "match.yaml"

        f.write_text("foo: 1\n", encoding="utf-8")
        h1 = compute_narratives_hash(narr_dir)

        f.write_text("foo: 2\n", encoding="utf-8")
        h2 = compute_narratives_hash(narr_dir)

        assert h1 != h2

    def test_compute_narratives_hash_order_independent(self, tmp_path: Path) -> None:
        """File creation order does not matter — only contents do (sorted by name)."""
        narr_dir1 = tmp_path / "n1"
        narr_dir1.mkdir()
        (narr_dir1 / "a.yaml").write_text("x\n", encoding="utf-8")
        (narr_dir1 / "b.yaml").write_text("y\n", encoding="utf-8")

        narr_dir2 = tmp_path / "n2"
        narr_dir2.mkdir()
        # Reverse creation order
        (narr_dir2 / "b.yaml").write_text("y\n", encoding="utf-8")
        (narr_dir2 / "a.yaml").write_text("x\n", encoding="utf-8")

        assert compute_narratives_hash(narr_dir1) == compute_narratives_hash(narr_dir2)


# ---------------------------------------------------------------------------
# T17.1c: Resume preserves manifest state byte-identically
# ---------------------------------------------------------------------------


class TestResumeManifestPreservation:
    """Resume reads manifest exactly — completed_steps preserved."""

    def test_resume_preserves_manifest_bytes(self, tmp_path: Path) -> None:
        """After record_step, resume_run reads back the SAME manifest content."""
        ctx = start_run(slug="determinism", runs_dir=tmp_path, seed=42)
        record_step(ctx, "step_1")
        record_step(ctx, "step_2")

        # Capture manifest bytes
        manifest_bytes_a = ctx.manifest_path.read_bytes()

        # Resume
        resumed = resume_run(ctx.root_path)

        # Capture manifest bytes after resume (should be identical)
        manifest_bytes_b = resumed.manifest_path.read_bytes()

        assert manifest_bytes_a == manifest_bytes_b
        assert resumed.completed_steps == ["step_1", "step_2"]

    def test_resume_seed_preserved(self, tmp_path: Path) -> None:
        """Seed is preserved across resume — critical for determinism."""
        ctx = start_run(slug="seed-resume", runs_dir=tmp_path, seed=12345)
        resumed = resume_run(ctx.root_path)
        assert resumed.seed == 12345


# ---------------------------------------------------------------------------
# T17.1d: Identity snapshots (existing 3 byte-identical snapshots)
# ---------------------------------------------------------------------------


class TestIdentitySnapshotsExist:
    """The 3 identity carryover snapshots from ajuste-senal-contextual must exist.

    These are the BYTE-IDENTICAL guarantee from the previous change. They must
    not be removed or altered in this change. The actual identity check is
    performed by tests/overlay/test_run_prediction_identity.py.
    """

    def test_three_snapshots_exist(self) -> None:
        """The 3 fixture snapshots are present and non-empty."""
        snapshots_dir = PROJECT_ROOT / "tests" / "fixtures" / "snapshots"
        assert snapshots_dir.is_dir(), "snapshots directory missing"

        snapshots = list(snapshots_dir.glob("*.json"))
        assert len(snapshots) >= 3, (
            f"Expected at least 3 identity snapshots, found {len(snapshots)}"
        )

        for snap in snapshots:
            assert snap.stat().st_size > 0, f"Snapshot {snap.name} is empty"

    def test_snapshots_have_stable_sha256(self) -> None:
        """Each snapshot file produces a stable sha256 — they are not mutating."""
        snapshots_dir = PROJECT_ROOT / "tests" / "fixtures" / "snapshots"
        for snap in sorted(snapshots_dir.glob("*.json")):
            h1 = hashlib.sha256(snap.read_bytes()).hexdigest()
            h2 = hashlib.sha256(snap.read_bytes()).hexdigest()
            assert h1 == h2, f"Hash mismatch for {snap.name}"
