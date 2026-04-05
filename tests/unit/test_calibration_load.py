"""Unit tests for calibration restoration in FoulPredictionEnsemble.load().

Spec ref: P1.1 — Calibration Restored on Load
- After ensemble.load(dir) when calibration.npz exists → calibration._is_fitted == True
- When calibration.npz absent → no error, _is_fitted == False
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "prediction_models"),
)

from src.models.calibration import OUCalibrationLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_calibration_files(directory: Path) -> None:
    """Write a minimal but valid calibration.npz + calibration.json.

    Args:
        directory: Target directory (must already exist).
    """
    line = 25.5
    key = f"line_{line:.1f}".replace(".", "_")
    X = np.array([0.2, 0.5, 0.8])
    y = np.array([0.18, 0.52, 0.79])
    np.savez(directory / "calibration.npz", **{f"{key}_X": X, f"{key}_y": y})
    meta = {"lines": [line], "calibrated_lines": [line]}
    with open(directory / "calibration.json", "w") as f:
        json.dump(meta, f)


def _write_minimal_ensemble_files(directory: Path) -> None:
    """Write the minimum artifact set for ensemble.load() to succeed.

    Creates structurally valid (but randomly-initialized) artifacts so that
    no real training data is needed.  Modeled after the save() contract in
    ensemble.py.

    Args:
        directory: Target directory (must already exist).
    """
    import torch
    from src.models.regression import FoulRegressionPredictor
    from src.models.anfis import ANFISFoulPredictor
    from src.models.gating_network import DynamicEnsembleWeighter
    from src.models.referee_gmm import RefereeProfiler

    # ---- PyTorch model state_dicts (default random init) ----
    reg = FoulRegressionPredictor()
    torch.save(reg.model.state_dict(), directory / "regression.pt")

    anfis = ANFISFoulPredictor()
    torch.save(anfis.model.state_dict(), directory / "anfis.pt")

    weighter = DynamicEnsembleWeighter()
    weighter.save(str(directory / "gating.pt"))

    # ---- bayes.npz (write arrays directly — no pre-fit NaiveBayes needed) ----
    # load() reads these keys; supply minimal but shape-compatible arrays.
    n_classes = 8
    n_clusters = 5
    bayes_arrays: dict[str, np.ndarray] = {
        "class_priors": np.ones(n_classes) / n_classes,
        "referee_cond_probs": np.ones((2, n_classes)) / n_classes,
        "breakpoints": np.array([0, 15, 20, 23, 26, 29, 32, 36, 40]),
        "foul_clusterer_committed_centroids": np.linspace(10, 16, n_clusters),
        "foul_clusterer_suffered_centroids": np.linspace(10, 16, n_clusters),
        "rank_clusterer_centroids": np.linspace(2, 18, n_clusters),
        "xfouls_clusterer_centroids": np.linspace(10, 16, n_clusters),
        "agg_clusterer_centroids": np.linspace(0.3, 0.8, n_clusters),
        "ref_delta_clusterer_centroids": np.linspace(-2, 2, n_clusters),
        "referee_avg_clusterer_centroids": np.linspace(22, 30, n_clusters),
        "ref_avg_cond_probs": np.ones((n_clusters, n_classes)) / n_classes,
    }
    np.savez(directory / "bayes.npz", **bayes_arrays)
    with open(directory / "bayes_meta.json", "w") as f:
        json.dump({"cond_probs_keys": []}, f)

    # ---- normalization.npz + normalization_meta.json ----
    # Use feature-count 1 — load() just assigns these arrays to attributes
    # without checking shape, so any non-empty array will work.
    np.savez(
        directory / "normalization.npz",
        reg_means=np.zeros(1),
        reg_stds=np.ones(1),
        anfis_mins=np.zeros(1),
        anfis_maxs=np.ones(1),
    )
    with open(directory / "normalization_meta.json", "w") as f:
        json.dump(
            {
                "bias_correction": 0.0,
                "variance_posthoc_scale": 1.0,
                "has_team_regressor": False,
            },
            f,
        )

    # ---- Referee directory ----
    RefereeProfiler().save(directory / "referee")


# ---------------------------------------------------------------------------
# Tests: OUCalibrationLayer.load() — pure unit
# ---------------------------------------------------------------------------


class TestOUCalibrationLayerLoad:
    """Direct unit tests for OUCalibrationLayer.load()."""

    def test_load_sets_is_fitted_true_when_file_present(self) -> None:
        """load() with a valid calibration.npz → _is_fitted == True.

        Scenario: Happy path (spec P1.1 — happy path).
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_calibration_files(tmp_path)

            cal = OUCalibrationLayer()
            assert cal._is_fitted is False  # sanity: starts unfitted

            cal.load(tmp_path / "calibration.npz")

            assert cal._is_fitted is True

    def test_load_leaves_is_fitted_false_when_file_absent(self) -> None:
        """load() when calibration.npz does not exist → no error, _is_fitted == False.

        Scenario: File absent (spec P1.1 — file absent).
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Do NOT write any calibration files

            cal = OUCalibrationLayer()
            cal.load(tmp_path / "calibration.npz")  # must not raise

            assert cal._is_fitted is False

    def test_load_populates_calibrators(self) -> None:
        """After load(), calibrators dict is populated with the saved line data."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_calibration_files(tmp_path)

            cal = OUCalibrationLayer()
            cal.load(tmp_path / "calibration.npz")

            assert 25.5 in cal._calibrators
            X, y = cal._calibrators[25.5]
            np.testing.assert_array_almost_equal(X, [0.2, 0.5, 0.8])
            np.testing.assert_array_almost_equal(y, [0.18, 0.52, 0.79])


# ---------------------------------------------------------------------------
# Tests: FoulPredictionEnsemble.load() — integration
# ---------------------------------------------------------------------------


class TestEnsembleLoadRestoresCalibration:
    """FoulPredictionEnsemble.load() must restore calibration state.

    Spec P1.1: After load(), calibration._is_fitted == True when
    calibration.npz is present in the checkpoint directory.
    """

    def test_load_with_calibration_sets_is_fitted_true(self) -> None:
        """ensemble.load(dir) + calibration.npz present → calibration._is_fitted == True.

        This is the RED-phase test: it FAILS with current code because load()
        does NOT call self.calibration.load().
        """
        from src.models.ensemble import FoulPredictionEnsemble

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_minimal_ensemble_files(tmp_path)
            _write_calibration_files(tmp_path)

            ensemble = FoulPredictionEnsemble()
            ensemble.load(tmp_path)

            assert ensemble.calibration._is_fitted is True, (
                "ensemble.load() must call self.calibration.load() "
                "to restore calibration state — currently missing."
            )

    def test_load_without_calibration_does_not_raise(self) -> None:
        """ensemble.load(dir) when calibration.npz absent → no error, _is_fitted == False.

        Scenario: regression guard — old checkpoint without calibration file.
        """
        from src.models.ensemble import FoulPredictionEnsemble

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_minimal_ensemble_files(tmp_path)
            # Deliberately do NOT write calibration files

            ensemble = FoulPredictionEnsemble()
            ensemble.load(tmp_path)  # must not raise

            assert ensemble.calibration._is_fitted is False
