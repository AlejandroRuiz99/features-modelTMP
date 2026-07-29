"""PromoteModelUseCase — gate check then registry promotion."""

from __future__ import annotations

from HWFP.core.domain.exceptions import ModelNotFound, PromotionGateFailed
from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import ModelManifest
from HWFP.core.ports.model_registry import ModelRegistry


class PromotionGates:
    """Pure policy: absolute threshold gates for model promotion.

    All three gates must pass; raises PromotionGateFailed listing failures.
    Gate names mirror the manifest's gates_passed field convention.
    """

    def check(self, manifest: ModelManifest) -> None:
        m = manifest.metrics_holdout
        failed: list[str] = []
        if m.nll >= 2.0:
            failed.append("nll_below_2.0")
        if m.brier >= 0.20:
            failed.append("brier_below_0.20")
        if m.calibration_ece >= 0.05:
            failed.append("ece_below_0.05")
        if failed:
            raise PromotionGateFailed(
                f"Manifest {manifest.model_id.value!r} failed gates: {failed}"
            )


class PromoteModelUseCase:
    def __init__(self, registry: ModelRegistry, gates: PromotionGates) -> None:
        self._registry = registry
        self._gates = gates

    def execute(self, model_id: ModelId) -> None:
        # 1. Find manifest (required for gates check before touching registry state)
        manifests = self._registry.list_manifests()
        manifest = next((m for m in manifests if m.model_id == model_id), None)
        if manifest is None:
            raise ModelNotFound(f"Model not found: {model_id.value!r}")
        # 2. Absolute threshold gate check — raises PromotionGateFailed on failure
        self._gates.check(manifest)
        # 3. Promote; registry handles relative comparison + prior-production archival
        self._registry.promote(model_id)
