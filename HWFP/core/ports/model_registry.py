"""ModelRegistry port — persist, promote, and retrieve prediction models."""

from __future__ import annotations

from typing import Protocol, Tuple

from HWFP.core.domain.model_id import ModelId
from HWFP.core.domain.model_manifest import ModelManifest
from HWFP.core.ports.foul_model import FoulModel


class ModelRegistry(Protocol):
    """Persist and manage the lifecycle of foul prediction models.

    Raises:
        NoProductionModelError: If load_production() is called with no production model.
        ModelNotFound: If promote() or archive() references an unknown model_id.
        PromotionGateFailed: If promotion gates check fails.
    """

    def load_production(self) -> FoulModel: ...

    def register(self, manifest: ModelManifest, model_blob: bytes) -> None: ...

    def promote(self, model_id: ModelId) -> None: ...

    def archive(self, model_id: ModelId) -> None: ...

    def list_manifests(self) -> Tuple[ModelManifest, ...]: ...
