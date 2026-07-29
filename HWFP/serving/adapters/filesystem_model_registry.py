"""FilesystemModelRegistry — ModelRegistry port adapter.

Loads FoulPredictionEnsemble from a checkpoints directory on disk and
exposes it as a FoulModel via load_production(). Read-only: register,
promote, and archive raise NotImplementedError.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

from HWFP.core.domain.exceptions import NoProductionModelError
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import HoldoutMetrics, ModelManifest
from HWFP.core.domain.model_status import ModelStatus

_REPO_ROOT = Path(__file__).parents[3]
_REQUIRED_FILES = ("gating.pt", "anfis.pt", "regression.pt", "bayes.npz")


def _ensure_prediction_models_path() -> None:
    p = str(_REPO_ROOT / "prediction_models")
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_manifest(checkpoints_dir: Path) -> ModelManifest:
    dir_name = checkpoints_dir.name or "ensemble"
    safe_name = dir_name.lower().replace(" ", "-")[:64]
    model_id = ModelId(value=safe_name if len(safe_name) >= 3 else "ensemble-v1")

    nll, brier, ece = 0.0, 0.0, 0.0
    cal_json = checkpoints_dir / "calibration.json"
    if cal_json.exists():
        with open(cal_json) as f:
            cal_data = json.load(f)
        nll = float(cal_data.get("nll", 0.0))
        brier = float(cal_data.get("brier", 0.0))
        ece = float(cal_data.get("ece", 0.0))

    trained_at = datetime.fromtimestamp(
        (checkpoints_dir / "gating.pt").stat().st_mtime
    )

    return ModelManifest(
        model_id=model_id,
        trained_at=trained_at,
        git_sha="unknown",
        dataset_hash="unknown",
        dataset_rows=0,
        metrics_holdout=HoldoutMetrics(nll=nll, brier=brier, calibration_ece=ece),
        gates_passed=(),
        status=ModelStatus.PRODUCTION,
    )


class FilesystemModelRegistry:
    """Loads the ensemble checkpoint from disk and implements ModelRegistry.

    Lazy-loads the ensemble weights on first call to load_production() so
    that construction is always fast (safe for DI containers and tests).
    """

    def __init__(self, checkpoints_dir: Path) -> None:
        self._checkpoints_dir = Path(checkpoints_dir)
        self._ensemble = None

    def _checkpoints_complete(self) -> bool:
        return self._checkpoints_dir.exists() and all(
            (self._checkpoints_dir / f).exists() for f in _REQUIRED_FILES
        )

    def _ensure_ensemble(self):
        if self._ensemble is None:
            _ensure_prediction_models_path()
            from src.models.ensemble import FoulPredictionEnsemble

            ensemble = FoulPredictionEnsemble()
            ensemble.load(self._checkpoints_dir)
            self._ensemble = ensemble
        return self._ensemble

    def load_production(self):
        if not self._checkpoints_complete():
            raise NoProductionModelError(
                f"No complete ensemble checkpoint found at: {self._checkpoints_dir}"
            )
        from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel

        return PyTorchFoulModel(ensemble=self._ensure_ensemble())

    def list_manifests(self) -> Tuple[ModelManifest, ...]:
        if not self._checkpoints_complete():
            return ()
        return (_build_manifest(self._checkpoints_dir),)

    def register(self, manifest: ModelManifest, model_blob: bytes) -> None:
        raise NotImplementedError("FilesystemModelRegistry is read-only")

    def promote(self, model_id: ModelId) -> None:
        raise NotImplementedError("FilesystemModelRegistry is read-only")

    def archive(self, model_id: ModelId) -> None:
        raise NotImplementedError("FilesystemModelRegistry is read-only")
