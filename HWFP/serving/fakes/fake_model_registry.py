"""FakeModelRegistry — in-memory model registry with full state machine. Zero I/O."""

from __future__ import annotations

from dataclasses import replace as _replace
from datetime import datetime

from HWFP.core.domain.exceptions import (
    ModelNotFound,
    NoProductionModelError,
    PromotionError,
)
from HWFP.core.domain.foul_pmf import FoulPMF
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import HoldoutMetrics, ModelManifest
from HWFP.core.domain.model_status import ModelStatus


class _InlineFoulModel:
    """Minimal FoulModel-protocol-satisfying object stored per registry entry."""

    _PMF: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05)
    _BIN_EDGES: tuple[int, ...] = (0, 15, 20, 22, 24, 26, 30, 40)

    def predict(self, features: tuple[float, ...]) -> FoulPMF:
        return FoulPMF(pmf=self._PMF, bin_edges=self._BIN_EDGES)


class FakeModelRegistry:
    """In-memory model registry with full state machine.

    State transitions: CANDIDATE → PRODUCTION → ARCHIVED.
    Promotion requires candidate metrics to beat current production
    (NLL lower OR Brier lower); raises PromotionError otherwise.
    Raises NoProductionModelError when load_production() is called with no production set.

    with_production_model() classmethod provides a pre-loaded production fixture
    for the golden e2e test.
    """

    def __init__(self) -> None:
        self._manifests: dict[str, ModelManifest] = {}
        self._models: dict[str, _InlineFoulModel] = {}
        self._production_id: str | None = None

    @classmethod
    def with_production_model(cls) -> FakeModelRegistry:
        """Return instance pre-loaded with one production model for golden e2e."""
        registry = cls()
        mid = ModelId("production-fixture-001")
        manifest = ModelManifest(
            model_id=mid,
            trained_at=datetime(2026, 6, 16, 10, 0, 0),
            git_sha="fixture",
            dataset_hash="hash-fixture-001",
            dataset_rows=100,
            metrics_holdout=HoldoutMetrics(nll=1.5, brier=0.15, calibration_ece=0.03),
            gates_passed=(),
            status=ModelStatus.PRODUCTION,
        )
        registry._manifests[mid.value] = manifest
        registry._models[mid.value] = _InlineFoulModel()
        registry._production_id = mid.value
        return registry

    def register(self, manifest: ModelManifest, model_blob: bytes) -> None:
        mid = manifest.model_id.value
        self._manifests[mid] = manifest
        self._models[mid] = _InlineFoulModel()

    def promote(self, model_id: ModelId) -> None:
        mid = model_id.value
        if mid not in self._manifests:
            raise ModelNotFound(f"Model not found: {mid!r}")
        candidate = self._manifests[mid]
        if self._production_id is not None:
            prod_m = self._manifests[self._production_id].metrics_holdout
            cand_m = candidate.metrics_holdout
            if not (cand_m.nll < prod_m.nll or cand_m.brier < prod_m.brier):
                raise PromotionError(
                    f"Candidate {mid!r} does not beat production — "
                    f"nll {cand_m.nll} vs {prod_m.nll}, "
                    f"brier {cand_m.brier} vs {prod_m.brier}"
                )
            prev = self._production_id
            self._manifests[prev] = _replace(
                self._manifests[prev], status=ModelStatus.ARCHIVED
            )
        self._production_id = mid
        self._manifests[mid] = _replace(candidate, status=ModelStatus.PRODUCTION)

    def load_production(self) -> _InlineFoulModel:
        if self._production_id is None:
            raise NoProductionModelError("No production model registered.")
        return self._models[self._production_id]

    def archive(self, model_id: ModelId) -> None:
        mid = model_id.value
        if mid not in self._manifests:
            raise ModelNotFound(f"Model not found: {mid!r}")
        self._manifests[mid] = _replace(
            self._manifests[mid], status=ModelStatus.ARCHIVED
        )

    def list_manifests(
        self, status: ModelStatus | None = None
    ) -> tuple[ModelManifest, ...]:
        result = tuple(self._manifests.values())
        if status is not None:
            result = tuple(m for m in result if m.status == status)
        return result
