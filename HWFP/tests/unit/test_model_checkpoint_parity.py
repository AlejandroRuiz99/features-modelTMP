"""Checkpoint byte-parity regression test (PR1 — models leaf package move).

Pins the sha256 of every file in the production ensemble checkpoint as it
existed at ``prediction_models/checkpoints/ensemble`` (repo HEAD 76c1122)
BEFORE the move to ``HWFP/models/checkpoints/ensemble``. Proves the move
preserved bytes exactly (spec: "Checkpoint bytes preserved").
"""

from __future__ import annotations

import hashlib

import pytest

from HWFP.models.paths import default_checkpoints_dir

# Pinned pre-move sha256 hashes, computed against
# prediction_models/checkpoints/ensemble at HEAD 76c1122 (before git mv).
_EXPECTED_SHA256: dict[str, str] = {
    "anfis.pt": "eb85654d2b38df96648819663700818e790c391384310200cca6e6d94d79d607",
    "bayes.npz": "6bddcd95688a02c700c57dc1fd23bd1877eba189b294287ab23ffaa700fffe5d",
    "bayes_meta.json": "7fc416c8740098ea217916ddea19f7485d395c4877bdd66e26172a31eaa43744",
    "calibration.json": "5825bb4fea68c40dde8c6e0d5c1b4ab1e83a9fe8091a542522a4d35c3d8f02df",
    "calibration.npz": "d7050b86ae98595dc251a8f2c469e5201cfdb55be6e31b36b48b1744bc6ae616",
    "config.json": "77cbfbf74cd17eae9da8957f2e3d4135fb05c627996ac6459d58210282b2ad65",
    "gating.pt": "f2dbba46f468fb54b2af077b43668140e3677516b0cfa96d552985481a69758e",
    "normalization.npz": "3cbce9d5a6cd0836b1e846579c041289aac4e61577b95424735abe2fc161da61",
    "normalization_meta.json": "e4cd2152983dcc74e86190f25675f49aaaeb6d8d91e06d08932c1dc89a2a39f2",
    "ratio_estimator.pt": "80d4d6cfe038808d384b885342c67b0a4ced31eb23b595481118774791cbda38",
    "referee/mode_selector.pt": "cb0f188e62ca7860ef8d806b7aac2ae9999bddf97f6d0d55a8212bc4afb201db",
    "referee/profiles.pkl": "c274fd4e155b0fc54121dd9c2e2db3ce251a333fb3e03f2818c6814e511fec37",
    "referee/profiles.pkl.bak.20260502_143201": "e0bf3d1cdedefec7096fe4e4269780f79c742fda10797e8efcd0e859ec9191b9",
    "regression.pt": "09a586fc59f76de948847f50c2c6cdd0d017ecb599db20d5b4d65b831b5067a0",
    "team_away.pt": "663d8f6e633179a0245debf28ea1ba12ba93710e860bc3ea4cfc26d80a6a615d",
    "team_home.pt": "ba7a8ff7b12371a3e958435b6bc912fbae534db26218b4a878da32f5b0aa20e8",
    "tuning_magic_constants.json": "1bf6ff231176f204ad09938fae8ecc44809c8bac7371175c5dcba801f825e0c7",
    "tuning_variance_knobs_2025_26.json": "28880ab4c9c82b8ef4f017130daeb568a15f2cbc3972681256a99e87892d63b7",
}


def _sha256_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCheckpointBytesPreserved:
    def test_all_pinned_files_present_and_unchanged(self):
        checkpoints_dir = default_checkpoints_dir()
        if not checkpoints_dir.exists():
            pytest.skip("Real checkpoints not available")

        for rel_path, expected_hash in _EXPECTED_SHA256.items():
            actual_path = checkpoints_dir / rel_path
            assert actual_path.exists(), f"missing checkpoint file: {rel_path}"
            assert _sha256_of(actual_path) == expected_hash, (
                f"checkpoint bytes changed during move: {rel_path}"
            )

    def test_no_extra_untracked_files_appeared(self):
        checkpoints_dir = default_checkpoints_dir()
        if not checkpoints_dir.exists():
            pytest.skip("Real checkpoints not available")

        actual_files = {
            str(p.relative_to(checkpoints_dir)).replace("\\", "/")
            for p in checkpoints_dir.rglob("*")
            if p.is_file()
        }
        assert actual_files == set(_EXPECTED_SHA256.keys())
